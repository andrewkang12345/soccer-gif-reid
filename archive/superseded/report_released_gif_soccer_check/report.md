# Released GIF model: soccer evaluation check

Checked 2026-09-12T17:53:03.392191+00:00.

**The authors’ trained CLIP-Proj/AP3D checkpoint has not been located, so its soccer accuracy has not been measured.**

The current GitHub tree is unchanged at `304370c107d9091bea47bb9c5432ac76020c24c3`, contains no weight files, and has no releases. The ReID README instructs users to produce projection/AP3D weights by training. A fresh read of the supplied Baidu share lists dataset/annotation archives and YOLOX detector checkpoint archives. The evaluator loads ReID weights separately from the detector; a detector checkpoint cannot substitute for them.

[Official ReID instructions](https://github.com/Wojak27/GIF/blob/main/reid/README.md) · [Official release page](https://github.com/Wojak27/GIF/releases) · [Fresh Baidu listing](baidu_current_listing.json)

## Local archive and checkpoint inspection — 2026-09-12

The supplied `data/mupnit_dataset.zip` **does include trained model weights**. Its two nested checkpoint ZIPs are `berzelius_yolox_nsvatrackv4_train.zip` and `berzelius_yolox_nsvatrackv4_trainval.zip`, also extracted under `data/MuPnit_YoloX_weights/YOLOX_NSVATrackv4/`.

Each contains four checkpoint snapshots (`best`, `latest`, `last_epoch`, `last_mosaic_epoch`), a training log and a TensorBoard event file. All eight extracted checkpoints were inspected with `torch.load(..., map_location="meta", weights_only=True)`. Each has root keys `start_epoch`, `model`, `optimizer` and 894 model state entries, comprising the YOLOX backbone and detection head. Representative shapes are `head.cls_preds.0.weight: [1, 320, 1, 1]`, `head.reg_preds.0.weight: [4, 320, 1, 1]`, and `head.obj_preds.0.weight: [1, 320, 1, 1]`. These are single-class player detection checkpoints, without a CLIP projection or AP3D ReID model. Both bundled training logs independently identify the YOLOX experiment, one detection class, depth 1.33 and width 1.25 (YOLOX-X).

The earlier answer should have explicitly distinguished the released detector weights from the missing ReID weights, and inspected these local archives before making the statement. This inspection confirms detector weights are present; it does not establish that trained ReID weights cannot exist elsewhere.

Both nested ReID dataset TARs were also enumerated directly: the full archive contains 396,376 JPEGs, one README and one JSON description file; the light archive contains 55,497 JPEGs, one README and one JSON description file. Neither contains model weights. The annotation ZIPs contain JSON annotations and a README. All eight inspected checkpoint paths and sizes match the two checkpoint ZIP inventories.

Evidence: [ZIP inventories](archive_contents.json) · [All eight checkpoint structures](checkpoint_contents.json) · [ReID TAR inventories](reid_tar_contents.json).

## Available baseline measured on real soccer

Ran GIF’s published image-only, vanilla CLIP ViT-L/14 baseline using official CLIP preprocessing (Resize + CenterCrop), normalized image embeddings and cosine matching. No projection weights, text attributes, OCR, added quality degradation or retraining. This does not evaluate the authors’ trained ReID models.

| Condition | Queries | Rank-1 | mAP |
|---|---:|---:|---:|
| Single crop | 511 | 4.11% | 16.06% |
| Four observations, GT identity grouping | 109 | 4.59% | 16.35% |

The image result is 21 correct out of 511 crops. There is one reference per identity, so image-gallery mAP equals mean reciprocal rank here. The ranking dump was independently recounted to verify both metrics.

## Data and limitations

- Real ARG_FRA_183303 soccer frames from the existing local diagnostic and 21 external roster portraits from `sports_reID/models/player_faces/`.
- 31 labeled images; 575 total annotations; 521 named-player boxes. Excluded 54 non-player labels and all 10 boxes in five contradictory frame/identity pairs before inference. No choice among contradictory boxes was made using recognition scores.
- Both teams are in the gallery. Full GT player bounding boxes are used, retaining original source pixels and saving crops losslessly. This does not evaluate detector/tracker performance.
- The 109 pooled examples use non-overlapping four-observation sequences with adjacent annotation sample indices and GT identity grouping; they cover 20 identities. These are sparse sampled observations, not McByte++ output or dense AP3D clips.
- Gallery names are inherited from filenames, not independently certified. Other query annotation mistakes may remain. This clip has been inspected previously for OCR, so this is a reused diagnostic rather than a fresh multi-match benchmark.
- Default CLIP center cropping may remove a player’s head/feet from a tall bbox. The available-baseline result cannot establish how the authors’ trained model would perform.

## Reproduce

```bash
cd /mnt/data/soccer_gif_reid
.venv/bin/python -m experiments.released_gif_soccer.prepare
.venv/bin/python -m experiments.released_gif_soccer.evaluate_clip_baseline
```

The frozen soccer input manifest is `data/arg_fra_roster_probe_v1/manifest.jsonl`. Obtain the authors’ trained projection or AP3D checkpoint to complete the requested model evaluation. No messages have been sent to the authors.
