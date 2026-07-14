"""Run a validation sweep to find the best image preprocessing config.

This script evaluates official PARSeq on the validation split under several
preprocessing variants and writes a ranked CSV plus the best config JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_DIR.parent
TRAIN_DIR = PIPELINE_DIR / "train_no_refinement"
PARSEQ_DIR = PIPELINE_DIR / "parseq"
for path in (TRAIN_DIR, PARSEQ_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from preprocessing import iter_named_configs, preprocess_plate_image  # noqa: E402
from strhub.data.module import SceneTextDataModule  # noqa: E402
from parseq_official_anpr_pipeline import (  # noqa: E402
    OfficialPARSeqANPRConfig,
    PlateCropDataset,
    collate_batch,
    create_official_parseq_model,
    evaluate,
    load_checkpoint,
)


def build_loader(data_root: Path, split: str, cfg: OfficialPARSeqANPRConfig, preprocess_cfg):
    base_transform = SceneTextDataModule.get_transform(tuple(cfg.img_size), augment=False)
    transform = T.Compose([T.Lambda(lambda image: preprocess_plate_image(image, preprocess_cfg)), base_transform])
    dataset = PlateCropDataset(data_root, split, transform, cfg.max_label_length)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


def run_sweep(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OfficialPARSeqANPRConfig(
        data_root=str(data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_label_length=args.max_label_length,
        refine_iters=args.refine_iters,
        decode_ar=args.decode_ar,
        preprocess=False,
        augment=False,
    )

    if args.checkpoint:
        model, _ckpt_cfg, _checkpoint = load_checkpoint(args.checkpoint, device=device)
    else:
        model = create_official_parseq_model(cfg, device=device)

    rows = []
    predictions = {}
    for preprocess_cfg in iter_named_configs(args.configs):
        loader = build_loader(data_root, args.split, cfg, preprocess_cfg)
        metrics, preds = evaluate(
            model,
            loader,
            device=device,
            split_name=f"{args.split}_{preprocess_cfg.name}",
            max_length=cfg.max_label_length,
        )
        row = {**preprocess_cfg.to_dict(), **metrics}
        rows.append(row)
        if args.save_predictions:
            predictions[preprocess_cfg.name] = preds
            preds.to_csv(output_dir / f"predictions_{preprocess_cfg.name}.csv", index=False)

    results = pd.DataFrame(rows).sort_values(["exact_acc", "char_acc"], ascending=False).reset_index(drop=True)
    results.to_csv(output_dir / "preprocessing_sweep_results.csv", index=False)
    best = results.iloc[0].to_dict()
    (output_dir / "best_preprocessing_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return results, best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "ocr_dataset_rescued_bbox_new"))
    parser.add_argument("--output-dir", default=str(PIPELINE_DIR / "outputs" / "preprocessing_best_config"))
    parser.add_argument("--checkpoint", default="", help="Optional official PARSeq ANPR checkpoint to evaluate.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--configs", nargs="*", default=None, help="Subset of preprocessing config names to evaluate.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-label-length", type=int, default=12)
    parser.add_argument("--refine-iters", type=int, default=0)
    parser.add_argument("--decode-ar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    results_df, best_config = run_sweep(parse_args())
    print(results_df[["name", "samples", "exact_acc", "char_acc", "cer"]])
    print(json.dumps(best_config, ensure_ascii=False, indent=2))
