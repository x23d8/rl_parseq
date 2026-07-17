"""Local keyboard-driven reviewer for the fresh Phase 8 holdout queue."""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_8_consensus_ppo.review_contract import (  # noqa: E402
    MINIMUM_FORMAL_SAMPLES,
    REVIEW_FIELDS,
    assess_review_rows,
    normalize_label,
    validate_final_label,
    validate_immutable_rows,
)


ORIGINAL_QUEUE = HERE / "fresh_holdout_review_queue_v2.csv"
REVIEWED_QUEUE = HERE / "fresh_holdout_review_queue_reviewed_v2.csv"
HISTORICAL_MANIFEST = ROOT / "outputs" / "phase3_controlled_aug_full_frozen_eval" / "dataset_manifest.csv"
OPENED_EXTERNAL_MANIFEST = (
    ROOT
    / "reinforcement_learning"
    / "phase_7_compact_multiscale_ppo"
    / "external_holdout_cropped"
    / "external_manifest.csv"
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader), list(reader.fieldnames or [])


def manifest_labels(path: Path) -> set[str]:
    rows, fields = read_csv(path)
    if "label" not in fields:
        raise ValueError(f"Manifest lacks label column: {path}")
    return {normalize_label(row["label"]) for row in rows if normalize_label(row["label"])}


class ReviewStore:
    def __init__(
        self,
        original_path: Path,
        reviewed_path: Path,
        excluded_labels: set[str] | None = None,
        minimum_samples: int = MINIMUM_FORMAL_SAMPLES,
    ):
        self.original_path = original_path.resolve()
        self.reviewed_path = reviewed_path.resolve()
        self.original, self.fields = read_csv(self.original_path)
        self.rows, reviewed_fields = read_csv(self.reviewed_path)
        if reviewed_fields != self.fields:
            raise ValueError("Reviewed queue schema/row count differs from immutable queue")
        validate_immutable_rows(self.original, self.rows)
        self.excluded_labels = excluded_labels or set()
        self.minimum_samples = minimum_samples
        self.lock = threading.Lock()

    def assessment(self) -> tuple[list[dict[str, str]], dict, list[dict]]:
        return assess_review_rows(
            self.original, self.rows, self.excluded_labels, self.minimum_samples
        )

    def progress(self) -> dict:
        return self.assessment()[1]

    def first_unreviewed(self) -> int:
        for index, row in enumerate(self.rows):
            if not row["review_decision"].strip():
                return index
        return 0

    def item(self, index: int) -> dict:
        if index < 0 or index >= len(self.rows):
            raise IndexError(index)
        row = self.rows[index]
        _, progress, statuses = self.assessment()
        return {
            "index": index,
            "source_sheet_row": row["source_sheet_row"],
            "source_id": row["source_id"],
            "current_label": row["current_extracted_character"],
            "normalized_label": row["normalized_label"],
            "source": row["source"],
            "source_cropped": row["source_cropped"],
            "input_transform": row["required_input_transform"],
            "decision": row["review_decision"],
            "corrected_label": row["corrected_label"],
            "eligibility": statuses[index],
            "progress": progress,
        }

    def update(self, index: int, decision: str, corrected_label: str = "") -> dict:
        decision = decision.strip().lower()
        if decision not in {"accepted", "corrected", "rejected"}:
            raise ValueError("Decision must be accepted, corrected, or rejected")
        if decision == "corrected":
            corrected_label = validate_final_label(corrected_label)
        if decision != "corrected":
            corrected_label = ""
        with self.lock:
            if index < 0 or index >= len(self.rows):
                raise IndexError(index)
            previous_decision = self.rows[index]["review_decision"]
            previous_label = self.rows[index]["corrected_label"]
            self.rows[index]["review_decision"] = decision
            self.rows[index]["corrected_label"] = corrected_label
            try:
                self._atomic_write()
            except Exception:
                self.rows[index]["review_decision"] = previous_decision
                self.rows[index]["corrected_label"] = previous_label
                raise
            next_index = self._next_unreviewed(index)
            return {"saved": True, "next_index": next_index, "progress": self.progress()}

    def _next_unreviewed(self, current: int) -> int:
        for offset in range(1, len(self.rows) + 1):
            index = (current + offset) % len(self.rows)
            if not self.rows[index]["review_decision"].strip():
                return index
        return current

    def _atomic_write(self) -> None:
        temporary = self.reviewed_path.with_suffix(self.reviewed_path.suffix + ".tmp")
        backup = self.reviewed_path.with_suffix(self.reviewed_path.suffix + ".bak")
        backup_temporary = backup.with_suffix(backup.suffix + ".tmp")
        previous_bytes = self.reviewed_path.read_bytes()
        with backup_temporary.open("wb") as destination:
            destination.write(previous_bytes)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(backup_temporary, backup)
        with temporary.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, self.reviewed_path)

    def plate_image(self, index: int) -> bytes:
        row = self.rows[index]
        path = Path(row["source_image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if row["required_input_transform"] == "crop_source_bounding_box":
                box = [int(round(float(value))) for value in json.loads(row["source_bounding_box"])]
                if len(box) != 4:
                    raise ValueError("Invalid bounding box")
                left, top = max(0, box[0]), max(0, box[1])
                right, bottom = min(image.width, box[2]), min(image.height, box[3])
                if right <= left or bottom <= top:
                    raise ValueError("Bounding box clips to an empty image")
                image = image.crop((left, top, right, bottom))
            image.thumbnail((1400, 700), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=95, subsampling=0)
            return output.getvalue()

    def context_path(self, index: int) -> Path:
        row = self.rows[index]
        visual_value = (row.get("visual_image_path") or "").strip()
        visual_path = Path(visual_value) if visual_value else None
        if visual_path is not None and visual_path.is_file():
            return visual_path
        source_path = Path(row["source_image_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        return source_path


HTML = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 8 Fresh Holdout Review</title>
<style>
body{margin:0;background:#101318;color:#e9eef5;font:16px system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:18px}
.top,.buttons{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.card{background:#191e26;border:1px solid #303744;border-radius:12px;padding:14px;margin:12px 0}
#plate{display:block;max-width:100%;max-height:55vh;margin:auto;image-rendering:auto;background:#080a0d}.context{max-width:100%;max-height:230px;display:block;margin:auto}
button,input{font:inherit;border-radius:8px;border:1px solid #465063;padding:10px;background:#222936;color:#fff}button{cursor:pointer}.accept{background:#176b3a}.reject{background:#872b34}.correct{background:#8b641c}
.label{font-size:30px;font-weight:700;letter-spacing:2px}.muted{color:#a9b3c2}.warning{color:#ffbf69;font-weight:650}.ok{color:#6fda9b}.bar{height:8px;background:#303744;border-radius:8px;overflow:hidden;flex:1;min-width:180px}.fill{height:100%;background:#3aa66b}kbd{background:#303744;padding:2px 6px;border-radius:4px}
</style></head><body><main>
<div class="top"><strong>Phase 8 review</strong><span id="progress"></span><div class="bar"><div class="fill" id="fill"></div></div><button onclick="go(-1)">←</button><input id="index" type="number" min="1" style="width:90px"><button onclick="jump()">Đi</button><button onclick="go(1)">→</button></div>
<div class="card"><img id="plate" alt="plate crop"></div>
<div class="card"><div class="label" id="label"></div><div id="meta" class="muted"></div><div id="eligibility" class="muted" style="margin-top:6px"></div><div class="buttons" style="margin-top:12px"><button class="accept" onclick="save('accepted')"><kbd>A</kbd> Accept</button><button class="reject" onclick="save('rejected')"><kbd>R</kbd> Reject</button><input id="corrected" placeholder="Corrected label"><button class="correct" onclick="save('corrected')"><kbd>C</kbd> Correct</button></div><div id="saved" class="muted" style="margin-top:8px"></div></div>
<details class="card"><summary>Ảnh ngữ cảnh</summary><img id="context" class="context" alt="source context"></details>
<p class="muted">Phím: <kbd>A</kbd> accept, <kbd>R</kbd> reject, <kbd>C</kbd> focus correction, <kbd>Enter</kbd> save correction, <kbd>←</kbd>/<kbd>→</kbd> chuyển ảnh. Không chạy OCR/PPO.</p>
</main><script>
let current=0,total=0,item=null;
async function load(i){let r=await fetch('/api/item?index='+i);if(!r.ok){alert(await r.text());return}item=await r.json();current=item.index;total=item.progress.total;document.querySelector('#index').value=current+1;document.querySelector('#index').max=total;document.querySelector('#label').textContent=item.current_label;document.querySelector('#corrected').value=item.corrected_label||item.current_label;document.querySelector('#meta').textContent=`row ${item.source_sheet_row} · ${item.source_id} · ${item.source} · ${item.input_transform} · decision: ${item.decision||'blank'}`;renderEligibility(item.eligibility);renderProgress(item.progress);let stamp=Date.now();document.querySelector('#plate').src='/plate?index='+current+'&v='+stamp;document.querySelector('#context').src='/context?index='+current+'&v='+stamp;document.querySelector('#saved').textContent='';fetch('/plate?index='+((current+1)%total));}
function renderEligibility(e){let el=document.querySelector('#eligibility');let messages={blank:'Chưa review',rejected:'Đã reject',eligible:'Hợp lệ cho holdout',excluded_label_overlap:'Label trùng dữ liệu lịch sử/external — tự động loại khỏi holdout',excluded_duplicate_label:'Trùng label với một ảnh đã review — tự động loại',invalid_final_label:'Label rỗng, có ký tự không hỗ trợ hoặc dài quá 12',invalid_decision:'Decision không hợp lệ'};el.textContent=messages[e.status]||e.status;el.className=e.status==='eligible'?'ok':(e.status==='blank'||e.status==='rejected'?'muted':'warning')}
function renderProgress(p){document.querySelector('#progress').textContent=`prefix ${p.reviewed_prefix}/${p.total} · eligible ${p.eligible_prefix} · excluded ${p.excluded_label_overlap+p.excluded_duplicate_label} · cần ${p.remaining_to_minimum}`;document.querySelector('#fill').style.width=(100*p.reviewed_prefix/p.total)+'%'}
async function save(decision){let corrected=document.querySelector('#corrected').value;let r=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:current,decision,corrected_label:corrected})});let data=await r.json();if(!r.ok){alert(data.error);return}renderProgress(data.progress);if(data.progress.formal_ready){let saved=document.querySelector('#saved');saved.textContent='Đã đủ 500 group hợp lệ. Dừng review và chạy phase8-finalize.';saved.className='ok';alert('Phase 8 đã formal-ready. Có thể dừng review và chạy phase8-finalize.');return}document.querySelector('#saved').textContent='Đã lưu';await load(data.next_index)}
function go(delta){load((current+delta+total)%total)}function jump(){let n=parseInt(document.querySelector('#index').value||'1',10);load(Math.max(0,Math.min(total-1,n-1)))}
document.addEventListener('keydown',e=>{let typing=document.activeElement===document.querySelector('#corrected');if(typing&&e.key==='Enter'){e.preventDefault();save('corrected');return}if(typing)return;if(e.key==='a'||e.key==='A')save('accepted');else if(e.key==='r'||e.key==='R')save('rejected');else if(e.key==='c'||e.key==='C'){document.querySelector('#corrected').focus();document.querySelector('#corrected').select()}else if(e.key==='ArrowLeft')go(-1);else if(e.key==='ArrowRight')go(1)});
fetch('/api/start').then(r=>r.json()).then(x=>load(x.index));
</script></body></html>"""


def make_handler(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _json(self, payload: dict, status: int = 200):
            value = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(value)))
            self.end_headers()
            self.wfile.write(value)

        def _index(self, query: dict[str, list[str]]) -> int:
            return int(query.get("index", ["0"])[0])

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    value = HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(value)))
                    self.end_headers()
                    self.wfile.write(value)
                elif parsed.path == "/api/start":
                    self._json({"index": store.first_unreviewed()})
                elif parsed.path == "/api/item":
                    self._json(store.item(self._index(query)))
                elif parsed.path == "/plate":
                    value = store.plate_image(self._index(query))
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(value)))
                    self.end_headers()
                    self.wfile.write(value)
                elif parsed.path == "/context":
                    path = store.context_path(self._index(query))
                    value = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(value)))
                    self.end_headers()
                    self.wfile.write(value)
                else:
                    self.send_error(404)
            except Exception as error:
                self._json({"error": str(error)}, 400)

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/api/decision":
                self.send_error(404)
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 8192)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._json(
                    store.update(
                        int(payload["index"]),
                        str(payload["decision"]),
                        str(payload.get("corrected_label", "")),
                    )
                )
            except Exception as error:
                self._json({"error": str(error)}, 400)

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-queue", default=str(ORIGINAL_QUEUE))
    parser.add_argument("--reviewed-queue", default=str(REVIEWED_QUEUE))
    parser.add_argument("--historical-manifest", default=str(HISTORICAL_MANIFEST))
    parser.add_argument("--opened-external-manifest", default=str(OPENED_EXTERNAL_MANIFEST))
    parser.add_argument("--minimum-samples", type=int, default=MINIMUM_FORMAL_SAMPLES)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true", help="Validate queues and render one plate without serving.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Review server may bind only to localhost")
    excluded_labels = manifest_labels(Path(args.historical_manifest)) | manifest_labels(
        Path(args.opened_external_manifest)
    )
    store = ReviewStore(
        Path(args.original_queue),
        Path(args.reviewed_queue),
        excluded_labels,
        args.minimum_samples,
    )
    if args.check:
        index = store.first_unreviewed()
        preview = store.plate_image(index)
        print(
            json.dumps(
                {
                    "ready": True,
                    "start_index": index,
                    "next_queue_number": index + 1,
                    "preview_bytes": len(preview),
                    "progress": store.progress(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Phase 8 reviewer: http://{args.host}:{args.port} — Ctrl+C to stop", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
