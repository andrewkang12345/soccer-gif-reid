# SoccerNet crop-to-web-photo training results

**Soccer training improved test mAP, but did not establish a reliable accuracy improvement.** The fresh head scored 18.84% mAP versus the NBA head’s 14.52%, while Rank-1 remained only 13.43% and below frozen CLIP’s 17.91%. Fine-tuning the NBA head retained epoch zero because no trained epoch improved validation mAP.

Training and held-out test scoring are complete. This is a small custom named-player retrieval task, not the standard action-restricted SoccerNet challenge.

| Model | Validation Rank-1 / mAP | Test Rank-1 / mAP | Correct test crops |
|---|---:|---:|---:|
| Frozen CLIP | 16.98% / 15.44% | 17.91% / 13.75% | 24/134 |
| NBA-trained projection | 9.43% / 15.22% | 11.94% / 14.52% | 16/134 |
| Soccer-trained fresh head | 15.09% / 20.53% | 13.43% / 18.84% | 18/134 |
| NBA head fine-tuned on soccer | 9.43% / 15.22% | 11.94% / 14.52% | 16/134 |

Validation uses 16 players / 53 broadcast crops; test uses 24 different players / 134 broadcast crops. Training uses 38 other players / 766 broadcast and replay crops across nine matches. Gallery references: 260 train, 112 validation, 163 test, four to eight screened web photos per player. Real roster IDs, exact names and source matches are disjoint across custom splits.

Soccer-trained fresh head: best epoch 4, stopped after 12 epochs. Epoch zero was eligible. The selected checkpoints depend only on validation mAP, not test results.
NBA head fine-tuned on soccer: best epoch 0, stopped after 8 epochs. Epoch zero was eligible. The selected checkpoints depend only on validation mAP, not test results.

Both models keep CLIP ViT-L/14 frozen and train the 790,272-parameter GIF projection head. Existing kit-invariance augmentation is retained, without additional image-quality degradation; sixteen deterministic augmented variants per training crop were cached. References remain clean.

All four test scores were independently recounted from saved full gallery rankings. Metric mAP counts every reference of the correct player as a positive. Neither team nor match restricts test candidate identities.

Limits: small and imbalanced query counts; validation covers only two matches; source-caption and automatically roster-mapped labels may contain residual errors; references vary in count and era. These results cannot establish production accuracy or isolate augmentation/data quantity as the cause. Test images were reviewed for data curation before scoring, but were not used for checkpoint selection. Argentina–France is not scored because several of its players occur in training.

[Preparation/training protocol](protocol.md) · [Full metrics](results.json) · [Independent recount](independent_metric_check.json) · [Dataset audit](../../data/soccernet_named_v1/final_audit.json) · [Gallery](../../data/soccernet_named_v1/gallery/)

Inference with either self-contained checkpoint uses the existing `experiments.ap3d_no_quality_v2.infer` module. Supply `--roster data/soccernet_named_v1/gallery/test` (or your own identity-subfolder directory) and `--query /path/to/crop.png`. It averages references per identity and returns cosine scores, not calibrated probabilities; that aggregation differs from the individual-photo benchmark above.

A paired player-cluster bootstrap (5,000 draws) puts the fresh-head mAP difference versus the NBA head at a 95% interval of −2.10 to +10.70 percentage points. This includes zero and ignores additional match correlation. The pilot therefore suggests a possible gain but does not verify that soccer supervision solves the earlier failure. [Bootstrap details](bootstrap_deltas.json).

Mapping audit: 159 of 451 actions accepted; 292 rejected as ambiguous/inconsistent. Of 1,053 selected mapped boxes, 1,038 were recovered, 55 failed segmentation, and 30 further crops were excluded with the two players lacking adequate verified references. Final total: 953 query images. Web gallery: 535 images. Source-title matches alone were insufficient for ambiguous namesakes, which were excluded during review.
