"""
Convert MOTChallenge detection GT (gt/gt.txt + img1/) into YOLO label files for Ultralytics val().

MOT17Det / MOT20Det layout (after unzip):
  <split>/<SEQ>/img1/000001.jpg
  <split>/<SEQ>/gt/gt.txt

Ultralytics expects each image path to contain an `images` segment; labels live in a parallel
`labels` folder (see ultralytics `img2label_paths`). This script therefore creates:

  <split>/<SEQ>/images  -> symlink/junction to `img1` (unless --no-images-link)
  <split>/<SEQ>/labels/000001.txt

Then point `data:` yaml at `.../<SEQ>/images` (see datasets/mot17det.yaml).

GT format (comma-separated, 9+ fields):
  frame, id, bb_left, bb_top, bb_width, bb_height, mark, class, visibility, ...

By default we keep MOT pedestrian class 1, skip mark==0 (ignored), and skip visibility < min-visibility.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[misc, assignment]

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[misc, assignment]


def _parse_gt_line(line: str) -> tuple[int, float, float, float, float, int, int, float] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 9:
        return None
    try:
        frame = int(float(parts[0]))
        bb_left = float(parts[2])
        bb_top = float(parts[3])
        bb_w = float(parts[4])
        bb_h = float(parts[5])
        mark = int(float(parts[6]))
        cls = int(float(parts[7]))
        vis = float(parts[8])
    except (ValueError, IndexError):
        return None
    return frame, bb_left, bb_top, bb_w, bb_h, mark, cls, vis


def _collect_gt_by_frame(gt_path: Path) -> dict[int, list[tuple[float, float, float, float, int, int, float]]]:
    """Map frame index -> list of raw rows (bbox + mark + cls + vis)."""
    by_frame: dict[int, list[tuple[float, float, float, float, int, int, float]]] = defaultdict(list)
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_gt_line(line)
            if parsed is None:
                continue
            frame, bb_left, bb_top, bb_w, bb_h, mark, cls, vis = parsed
            by_frame[frame].append((bb_left, bb_top, bb_w, bb_h, mark, cls, vis))
    return by_frame


def _frame_from_name(name: str) -> int | None:
    stem = Path(name).stem
    if stem.isdigit():
        return int(stem)
    m = re.match(r"^0*(\d+)$", stem)
    return int(m.group(1)) if m else None


def _list_frames(img_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in sorted(img_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        fid = _frame_from_name(p.name)
        if fid is None:
            continue
        out.append((fid, p))
    return out


def _image_size(path: Path) -> tuple[int, int]:
    if Image is not None:
        with Image.open(path) as im:
            return im.size
    if cv2 is not None:
        im = cv2.imread(str(path))
        if im is None:
            raise RuntimeError(f"Could not read image: {path}")
        h, w = im.shape[:2]
        return w, h
    raise SystemExit(
        "Need Pillow or OpenCV to read image sizes. Install one of: pip install pillow  OR  pip install opencv-python"
    )


def _bbox_to_yolo_line(
    left: float,
    top: float,
    w: float,
    h: float,
    iw: int,
    ih: int,
    class_id: int,
) -> str:
    x_c = (left + w / 2.0) / float(iw)
    y_c = (top + h / 2.0) / float(ih)
    wn = w / float(iw)
    hn = h / float(ih)
    # clip to [0, 1]
    x_c = min(max(x_c, 0.0), 1.0)
    y_c = min(max(y_c, 0.0), 1.0)
    wn = min(max(wn, 0.0), 1.0)
    hn = min(max(hn, 0.0), 1.0)
    return f"{class_id} {x_c:.6f} {y_c:.6f} {wn:.6f} {hn:.6f}"


def _ensure_images_link(seq_dir: Path, img1: Path, *, no_link: bool) -> Path:
    """Return path to .../images used in yaml (symlink to img1 when possible)."""
    images = seq_dir / "images"
    if images.exists():
        if images.is_symlink() or images.is_dir():
            return images
    if no_link:
        return img1

    if sys.platform == "win32":
        # Junction (/J) usually works without admin for local dirs; symlinks often need Developer Mode.
        if not images.exists():
            try:
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(images), str(img1.resolve())],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                pass
        if not images.exists():
            try:
                os.symlink(img1, images, target_is_directory=True)
            except OSError:
                pass
    else:
        try:
            if not images.exists():
                os.symlink(os.path.relpath(img1, seq_dir), images, target_is_directory=True)
        except OSError:
            pass

    if images.exists():
        return images

    print(
        f"Warning: could not create 'images' link in {seq_dir}; use img1 directly in yaml "
        f"(Ultralytics may still require a path containing 'images' — see docs) or create manually:\n"
        f'  Windows (cmd, admin optional for /J): mklink /J "{images}" "{img1}"',
        file=sys.stderr,
    )
    return img1


def _filter_box(
    mark: int,
    cls: int,
    vis: float,
    *,
    classes: set[int],
    min_visibility: float,
    require_mark_positive: bool,
) -> bool:
    if cls not in classes:
        return False
    if require_mark_positive and mark == 0:
        return False
    if vis < min_visibility:
        return False
    return True


def convert_sequence(
    seq_dir: Path,
    *,
    classes: set[int],
    min_visibility: float,
    require_mark_positive: bool,
    yolo_class_id: int,
    dry_run: bool,
    no_images_link: bool,
) -> tuple[int, int]:
    """
    Write labels into seq_dir/labels. Ensure seq_dir/images -> img1.
    Returns (num_label_files_written, num_boxes_kept).
    """
    img1 = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    if not img1.is_dir():
        raise FileNotFoundError(f"Missing img1: {img1}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing gt: {gt_path}")

    frames_imgs = _list_frames(img1)
    if not frames_imgs:
        raise RuntimeError(f"No images in {img1}")

    by_frame = _collect_gt_by_frame(gt_path)

    labels_dir = seq_dir / "labels"
    if not dry_run:
        labels_dir.mkdir(parents=True, exist_ok=True)

    _ensure_images_link(seq_dir, img1, no_link=no_images_link)

    boxes_kept = 0
    files_written = 0

    for fid, img_path in frames_imgs:
        iw, ih = _image_size(img_path)

        lines_out: list[str] = []
        for left, top, bw, bh, mark, cls, vis in by_frame.get(fid, []):
            if not _filter_box(
                mark, cls, vis,
                classes=classes,
                min_visibility=min_visibility,
                require_mark_positive=require_mark_positive,
            ):
                continue
            if bw <= 0 or bh <= 0:
                continue
            lines_out.append(
                _bbox_to_yolo_line(left, top, bw, bh, iw, ih, yolo_class_id)
            )
            boxes_kept += 1

        out_txt = labels_dir / f"{fid:06d}.txt"
        if not dry_run:
            out_txt.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
        files_written += 1

    return files_written, boxes_kept


def discover_sequences(mot_root: Path, splits: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for sp in splits:
        root = mot_root / sp
        if not root.is_dir():
            continue
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir():
                continue
            if (seq_dir / "gt" / "gt.txt").is_file() and (seq_dir / "img1").is_dir():
                found.append(seq_dir)
    return found


def main() -> None:
    p = argparse.ArgumentParser(description="MOT Det gt.txt -> YOLO labels for Ultralytics")
    p.add_argument(
        "--mot-root",
        type=Path,
        required=True,
        help="Root folder that contains train/ and/or test/ (e.g. P:/MOT17Det)",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Which top-level splits to scan (default: train only; test usually has no public GT)",
    )
    p.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional sequence folder names to include (e.g. MOT17-02). Default: all with gt+img1.",
    )
    p.add_argument(
        "--classes",
        default="1",
        help="Comma-separated MOT class IDs to keep (default: 1=pedestrian)",
    )
    p.add_argument("--min-visibility", type=float, default=0.0, help="Min visibility in [0,1] (default: 0)")
    p.add_argument(
        "--require-mark-positive",
        action="store_true",
        help="Skip rows with 7th column mark==0 (ignored GT)",
    )
    p.add_argument("--yolo-class-id", type=int, default=0, help="YOLO class index written (default: 0 for single-class)")
    p.add_argument("--dry-run", action="store_true", help="Parse only; do not write files")
    p.add_argument(
        "--no-images-link",
        action="store_true",
        help="Do not create images->img1 link (you must arrange an `images` path for Ultralytics)",
    )
    args = p.parse_args()

    class_set = {int(x.strip()) for x in args.classes.split(",") if x.strip()}

    mot_root = args.mot_root.resolve()
    seq_dirs = discover_sequences(mot_root, args.splits)
    if args.sequences:
        allow = set(args.sequences)
        seq_dirs = [s for s in seq_dirs if s.name in allow]

    if not seq_dirs:
        print(f"No sequences with gt/gt.txt + img1 under {mot_root} / {args.splits}", file=sys.stderr)
        sys.exit(1)

    total_files = 0
    total_boxes = 0
    for seq_dir in seq_dirs:
        try:
            nw, nb = convert_sequence(
                seq_dir,
                classes=class_set,
                min_visibility=args.min_visibility,
                require_mark_positive=args.require_mark_positive,
                yolo_class_id=args.yolo_class_id,
                dry_run=args.dry_run,
                no_images_link=args.no_images_link,
            )
        except Exception as e:
            print(f"[skip] {seq_dir}: {e}", file=sys.stderr)
            continue
        total_files += nw
        total_boxes += nb
        print(f"OK {seq_dir.name}: labels={nw}, boxes={nb}")

    print(f"Done. Sequences={len(seq_dirs)}, label files={total_files}, boxes={total_boxes}")
    if args.dry_run:
        print("(dry-run: no files written)")


if __name__ == "__main__":
    main()
