"""Run the locked PPO restoration agent on a new cropped plate image."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "train_no_refinement", ROOT / "parseq", ROOT / "preprocessing_best_config"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from preprocessing_best_config.find_best_preprocessing_config import load_notebook_checkpoint  # noqa: E402
from train_no_refinement.parseq_official_anpr_pipeline import normalize_plate_text  # noqa: E402
from rl_restoration.actions import DEFAULT_ACTIONS  # noqa: E402
from rl_restoration.features import image_quality_features, parseq_state_features  # noqa: E402
from rl_restoration.policy import RewardRouter  # noqa: E402
from rl_restoration.ppo_policy import RestorationActorCritic  # noqa: E402
from rl_restoration.sequential_env import MAX_PLATE_LENGTH, encode_predictions  # noqa: E402


class PPORestorationRuntime:
    def __init__(self, ppo_checkpoint: Path, ocr_checkpoint: Path, device: str = "", refine_iters: int = 2):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.refine_iters = int(refine_iters)
        self.actions = DEFAULT_ACTIONS
        self.action_names = [action.name for action in self.actions]
        self.ocr, self.ocr_cfg, _ = load_notebook_checkpoint(ocr_checkpoint, self.device, self.refine_iters)
        self.checkpoint = torch.load(ppo_checkpoint, map_location="cpu", weights_only=False)
        if self.checkpoint.get("candidate_summary", False):
            raise ValueError(
                "This checkpoint needs all-candidate observations; use a standard locked PPO checkpoint for runtime."
            )
        if self.checkpoint["action_names"] != self.action_names:
            raise ValueError("PPO checkpoint action space differs from runtime registry")
        self.policy = RestorationActorCritic(
            self.checkpoint["input_dim"],
            len(self.actions),
            self.checkpoint["hidden_dim"],
            self.checkpoint["dropout"],
            self.checkpoint["prior_offset"],
            self.checkpoint["prior_scale"],
        ).to(self.device)
        self.policy.load_state_dict(self.checkpoint["model_state_dict"])
        self.policy.eval()
        teacher_checkpoint = torch.load(self.checkpoint["teacher_router"], map_location="cpu", weights_only=False)
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher = RewardRouter(
            teacher_checkpoint["input_dim"],
            len(self.actions),
            teacher_checkpoint["hidden_dim"],
            teacher_checkpoint["dropout"],
        ).to(self.device)
        self.teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
        self.teacher.eval()

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        resized = TF.resize(image.convert("RGB"), list(self.ocr_cfg.img_size), interpolation=InterpolationMode.BICUBIC)
        tensor = TF.normalize(TF.to_tensor(resized), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def _ocr_view(self, original: Image.Image, action_index: int, collect_deep: bool = False) -> dict:
        restored = self.actions[action_index].apply(original)
        tensor = self._tensor(restored)
        logits = self.ocr(tensor, max_length=self.ocr_cfg.max_label_length)
        probabilities = logits.softmax(-1)
        predictions, token_probabilities = self.ocr.tokenizer.decode(probabilities)
        prediction = normalize_plate_text(predictions[0])
        token_values = token_probabilities[0]
        confidence = float(token_values.prod().item())
        normalized_confidence = math.exp(math.log(max(confidence, 1e-12)) / max(len(prediction) + 1, 1))
        result = {
            "action_index": action_index,
            "action": self.action_names[action_index],
            "prediction": prediction,
            "confidence": confidence,
            "normalized_confidence": normalized_confidence,
            "restored_size": list(restored.size),
        }
        if collect_deep:
            deep = parseq_state_features(self.ocr, tensor, [prediction], logits).cpu().numpy()
            quality = image_quality_features(original)[None, :]
            result["base_features"] = np.concatenate((deep, quality), axis=1).astype(np.float32)
        return result

    def _action_observation(self, view: dict, baseline_prediction: str) -> np.ndarray:
        encoded = encode_predictions(np.asarray([[view["prediction"]]], dtype=str))[0, 0]
        return np.concatenate(
            (
                np.asarray(
                    [
                        view["normalized_confidence"],
                        len(view["prediction"]) / MAX_PLATE_LENGTH,
                        float(view["prediction"] != baseline_prediction),
                    ],
                    dtype=np.float32,
                ),
                encoded,
            )
        )

    def _state(self, base: np.ndarray, view: dict, baseline_prediction: str, action_index: int, step: int):
        action_one_hot = np.zeros(len(self.actions), dtype=np.float32)
        action_one_hot[action_index] = 1.0
        state = np.concatenate(
            (base, self._action_observation(view, baseline_prediction), action_one_hot, [float(step)])
        ).astype(np.float32)
        return torch.from_numpy(state).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def run(self, image: Image.Image) -> dict:
        original = image.convert("RGB")
        baseline = self._ocr_view(original, 0, collect_deep=True)
        raw_features = baseline.pop("base_features")
        standardized = (raw_features - self.checkpoint["feature_mean"]) / self.checkpoint["feature_std"]
        teacher_standardized = (
            raw_features - self.teacher_checkpoint["feature_mean"]
        ) / self.teacher_checkpoint["feature_std"]
        teacher_rewards = self.teacher(torch.from_numpy(teacher_standardized.astype(np.float32)).to(self.device))
        base = np.concatenate((standardized, teacher_rewards.cpu().numpy()), axis=1)[0]

        logits0, _ = self.policy(self._state(base, baseline, baseline["prediction"], 0, 0))
        best0 = int(logits0.argmax(dim=1).item())
        first_gain = float((logits0[0, best0] - logits0[0, 0]).item())
        first = best0 if best0 != 0 and first_gain >= float(self.checkpoint["first_margin"]) else 0
        if first == 0:
            return {
                "prediction": baseline["prediction"],
                "confidence": baseline["confidence"],
                "first_action": "stop_baseline",
                "final_action": "stop_baseline",
                "revised": False,
                "baseline": baseline,
                "intermediate": baseline,
                "final": baseline,
            }

        intermediate = self._ocr_view(original, first)
        logits1, _ = self.policy(
            self._state(base, intermediate, baseline["prediction"], first, 1)
        )
        best1 = int(logits1.argmax(dim=1).item())
        revise_gain = float((logits1[0, best1] - logits1[0, first]).item())
        final_index = best1 if revise_gain >= float(self.checkpoint["revise_margin"]) else first
        final = intermediate if final_index == first else baseline if final_index == 0 else self._ocr_view(original, final_index)
        return {
            "prediction": final["prediction"],
            "confidence": final["confidence"],
            "first_action": self.action_names[first],
            "final_action": self.action_names[final_index],
            "revised": final_index != first,
            "baseline": baseline,
            "intermediate": intermediate,
            "final": final,
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument(
        "--ppo-checkpoint",
        default=str(ROOT / "outputs/rl_restoration/ppo_prior_seed_123/best_ppo_restoration_policy.pt"),
    )
    parser.add_argument(
        "--ocr-checkpoint",
        default=str(ROOT / "outputs/rl_restoration/parseq_ppo_hard_curriculum/best_parseq_rl_policy_mixture.pt"),
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--refine-iters", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runtime = PPORestorationRuntime(
        Path(args.ppo_checkpoint), Path(args.ocr_checkpoint), args.device, args.refine_iters
    )
    with Image.open(args.image) as opened:
        result = runtime.run(opened.convert("RGB"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
