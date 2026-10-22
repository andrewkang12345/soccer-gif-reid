# Soccer evaluation log — 2026-09-12

Our v2 CLIP-Proj and AP3D were trained/evaluated on NBA data. This session added real-soccer checkpoint evaluation on the existing Argentina–France diagnostic and prepared SoccerNet queries for the requested eight-web-photo-per-player test.

The SoccerNet name/roster mapping and reference gallery are still missing; no SoccerNet retrieval accuracy is claimed. Both trained models have encoded all 1,144 prepared four-frame queries. Training and OCR experiments remain unchanged.

See [the full report](soccernet_web_gallery_v1/report.md) and [machine-readable status](soccernet_web_gallery_v1/status.json).

## Expanded Argentina–France gallery

Matched the actual training manifest with 8 photos/player (168 total), not 200. Fixed 109 queries and original v2 checkpoints; no retraining or added degradation. Gallery curated and frozen before scoring.

- clip: Rank-1 12.84%, mAP 13.21%; 14/109 correct.
- ap3d: Rank-1 6.42%, mAP 8.14%; 7/109 correct.
- base: Rank-1 5.50%, mAP 8.54%; 6/109 correct.

[Expanded-gallery report](arg_fra_expanded_gallery_v1/report.md). Independent ranked-output recount passed. SoccerNet named mapping remains unresolved; no new SoccerNet score claimed.

Both trained models remain substantially worse than NBA clean validation: CLIP-Proj −11.30 pp Rank-1 / −10.39 pp mAP; AP3D −5.03 / −4.35 pp. More gallery variety did not close the transfer gap.

## Diagnosis verification — 2026-09-13

New training-identity retrieval and saved-history/error audit support a generalization gap, but do not causally verify missing soccer pairs as the main problem. See [audit](diagnosis_audit_v1/report.md). No training or augmentation changes were made.
