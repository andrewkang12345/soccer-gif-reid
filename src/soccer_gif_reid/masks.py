from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .augment import BroadcastAugment, load_masks
from .common import atomic_json, atomic_write, digest_file, load_manifest, read_rgb, seed_for, write_manifest

MASK_VERSION = "central_person_padding_v2"


def select_person(result, shape):
    if result.masks is None:
        raise ValueError("No person segmentation")
    h, w = shape
    candidates = []
    for box, polygon, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.masks.xy, result.boxes.conf.cpu().numpy()):
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
        fraction = (mask > 127).mean()
        x1, y1, x2, y2 = box
        central = x1 <= w * 0.5 <= x2 and y1 <= h * 0.5 <= y2
        if central and 0.03 <= fraction <= 0.99:
            score = float(confidence) * (y2 - y1) / h
            candidates.append((score, mask))
    if not candidates:
        raise ValueError("No plausible central-player segmentation; manual mask required")
    return max(candidates, key=lambda x: x[0])[1]


def generate_person_mask(model, im, device):
    """Retry tight crops with context padding while preserving original pixels.

    Both attempts use actual predicted person polygons. Never replace a failed
    segmentation with a rectangle or an assumed human silhouette.
    """
    w, h = im.size
    a = np.asarray(im)
    edges = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    color = tuple(np.median(edges, axis=0).astype(int).tolist())
    px, py = max(8, round(w * 0.4)), max(8, round(h * 0.15))
    attempts = [(im, 0, 0, 0.15), (ImageOps.expand(im, (px, py, px, py), fill=color), px, py, 0.10)]
    errors = []
    for canvas, left, top, confidence in attempts:
        result = model.predict(np.array(canvas)[..., ::-1], classes=[0], device=device,
                               imgsz=640, conf=confidence, retina_masks=True, verbose=False)[0]
        try:
            mask = select_person(result, (canvas.height, canvas.width))[top:top + h, left:left + w]
            if not 0.03 <= (mask > 127).mean() <= 0.99:
                raise ValueError("Implausible person area after removing padding")
            return mask, "padded" if left else "original"
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError("No usable segmentation after original/padded attempts: " + "; ".join(errors))


def main():
    p = argparse.ArgumentParser(description="Generate query person masks, with explicit failure report and augmentation preview")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--weights", required=True, help="Person segmentation weights, e.g. yolo11m-seg.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preview", type=Path, default=Path("reports/augmentation_preview.jpg"))
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Refusing to overwrite {args.output}")
    from ultralytics import YOLO
    model = YOLO(args.weights)
    weight_path = Path(getattr(model, "ckpt_path", args.weights))
    weight_hash = digest_file(weight_path)
    rows = load_manifest(args.manifest)
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    failures, previews, methods = [], [], {}
    for i, row in enumerate(rows):
        if row["domain"] != "query":
            continue
        im = read_rgb(row["path"])
        key = digest_file(row["path"])
        row["sha256"] = key
        path = args.mask_dir.resolve() / f"{MASK_VERSION}_{weight_hash[:12]}_{key}.png"
        try:
            if row.get("person_mask"):
                load_masks(row, im.size)
            else:
                if not path.exists():
                    mask, method = generate_person_mask(model, im, args.device)
                    atomic_write(path, lambda p: Image.fromarray(mask).save(p, format="PNG"))
                    methods[method] = methods.get(method, 0) + 1
                row["person_mask"] = str(path)
                load_masks(row, im.size)
            if len(previews) < 12:
                masks = load_masks(row, im.size)
                aug = BroadcastAugment("heavy")(im, masks, np.random.default_rng(seed_for(1701, key)))
                panel = Image.new("RGB", (256, 244), (240, 240, 240))
                for x, src in [(0, im), (128, aug)]:
                    src = src.copy()
                    src.thumbnail((128, 220))
                    panel.paste(src, (x + (128 - src.width) // 2, 20))
                ImageDraw.Draw(panel).text((4, 3), f"ID {row['pid']} | original / degraded", fill="black")
                previews.append(panel)
        except (ValueError, OSError) as exc:
            failures.append({"path": row["path"], "error": str(exc)})
        if i % 100 == 0:
            print(f"Masks: {i}/{len(rows)}, failures={len(failures)}", flush=True)
    report = {"weights": args.weights, "weights_sha256": weight_hash, "mask_version": MASK_VERSION,
              "generated_methods": methods, "failures": failures, "query_images": sum(r["domain"] == "query" for r in rows),
              "clothing_masks": "Provided parsing masks, else conservative person/geometry/skin intersection"}
    atomic_json(args.output.with_suffix(".masks.json"), report)
    # Preserve successes for repair/resume, but training will reject unresolved masks.
    write_manifest(args.output, rows)
    if previews:
        sheet = Image.new("RGB", (256 * 3, 244 * ((len(previews) + 2) // 3)), "white")
        for i, panel in enumerate(previews):
            sheet.paste(panel, ((i % 3) * 256, (i // 3) * 244))
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.preview)
    if failures:
        raise SystemExit(f"{len(failures)} masks need repair; see {args.output.with_suffix('.masks.json')}")


if __name__ == "__main__":
    main()
