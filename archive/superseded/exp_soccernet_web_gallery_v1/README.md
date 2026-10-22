# SoccerNet broadcast queries to a named web gallery

This isolated experiment evaluates the existing, validation-selected `ap3d_no_quality_v2` CLIP-Proj and AP3D checkpoints. It performs no training and does not select weights or preprocessing on soccer results.

`prepare_soccernet.py` selects three official test clips per match with a fixed hash, and extracts up to eight disjoint four-frame samples per numerically labeled GT track. Crops retain the complete annotated box, source resolution and pixel values, and are saved as PNG. Four frames are separated by four source-frame indices (0.48 seconds total at 25 fps). Unknown jersey labels cannot serve as identity ground truth and are excluded. Frames are not filtered by OCR, legibility, visual quality, or model scores.

The local GSR annotations have anonymous match IDs and clip-relative left/right team labels. A key such as `SNGS-144:left:14` is an anonymous annotation identity, **not a verified player name**. Do not assume a left/right label identifies the same club across clips or halves. The name/club fields in `data/soccernet_web_gallery_v1/identity_mapping_required.csv` deliberately remain empty until supported by match/roster metadata.

The target is exactly **eight distinct external web reference photos per named player**, matching the eight individual references per identity in the NBA test. Photos must have traceable source URLs and verified player labels; query crops and variants of them cannot become gallery images. Multiple sizes or crops of one photograph do not count as different reference photos. Do not select photos by the tested model's similarity scores. Near-duplicate and content review is required in addition to the evaluator's exact-byte duplicate check. No named SoccerNet gallery is currently claimed to exist.

Prepare original-quality anonymous queries:

```bash
.venv/bin/python -m experiments.soccernet_web_gallery_v1.prepare_soccernet
```

After the name mapping and web photos are available, create a query JSONL with `query_id`, verified `pid`, and four distinct image `paths`, and a gallery JSONL with `pid`, local image `path`, and image/source URLs. The same manifests must be used for both models:

```bash
.venv/bin/python -m experiments.soccernet_web_gallery_v1.evaluate \
  --queries data/soccernet_web_gallery_v1/named_queries.jsonl \
  --gallery data/soccernet_web_gallery_v1/gallery.jsonl \
  --checkpoint runs/ap3d_no_quality_v2/clip/best.pt \
  --output reports/soccernet_web_gallery_v1/clip
```

Replace the checkpoint and output model name with `ap3d` for the second model. `--base` on the CLIP checkpoint evaluates its frozen pretrained visual backbone with the same letterbox preprocessing and frame pooling. Without `--gallery`, the evaluator only caches query features and explicitly produces **no accuracy metric**.

Report both image-gallery Rank-1/mAP (all eight same-player references are positives) and roster-prototype Rank-1/MRR (reference embeddings averaged per player). Cosine scores are not calibrated probabilities. This diagnostic uses GT crops and track grouping, so it does not assess McByte++ identity switches or end-to-end production accuracy.

The `arg_fra_auxiliary` outputs are a separate reused diagnostic: 109 four-observation sequences and 21 existing web portraits. They use one portrait per player and cannot substitute for the requested SoccerNet/eight-photo test.
