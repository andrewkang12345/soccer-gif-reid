# Argentina–France expanded-gallery evaluation

**Both trained models remain substantially worse than NBA validation.** CLIP-Proj is 11.30 percentage points lower in Rank-1 and 10.39 points lower in mAP; AP3D is 5.03 and 4.35 points lower. Expanding reference count and variety did not solve the transfer problem.

Used **8 reference photographs per player**, matching the manifest used by our actual training run: 944 references / 118 training identities. This is not a 200-image experiment. The full raw MuPNIT gallery averages 144.32 images per identity (median 173), but those were not all used by our training run.

The frozen gallery contains **168 photos / 21 players**. Evaluation reused exactly the same **109 four-observation queries / 20 players**, checkpoints and checkpoint-native letterbox preprocessing as the previous Argentina–France diagnostic. No retraining or added image-quality degradation. Query features were reused only after verifying checkpoint and query-manifest hashes.

| Model | Soccer Rank-1 | Soccer mAP | NBA clean validation Rank-1 | NBA clean validation mAP | Soccer minus NBA (R1 / mAP, pp) |
|---|---:|---:|---:|---:|---:|
| Our CLIP-Proj | 12.84% | 13.21% | 24.15% | 23.60% | -11.30 / -10.39 |
| Our AP3D | 6.42% | 8.14% | 11.46% | 12.49% | -5.03 / -4.35 |
| Frozen CLIP (matched preprocessing) | 5.50% | 8.54% | — | — | — |

These are individual-photo retrieval metrics, with all eight photos of the correct player counted as positives for AP, matching the NBA metric definition. NBA validation has 323 queries, 42 identities and 333 references (41 identities with eight, one with five); the soccer test has fewer distractor identities.

For deployment-style ranking using one mean reference embedding per identity, Rank-1 is clip: 8.26% (9/109), ap3d: 4.59% (5/109), base: 2.75% (3/109). This is a separate aggregation rule, not the NBA validation metric.

The previous one-portrait gallery gave CLIP-Proj 12.84% Rank-1 and AP3D 11.01%; frozen CLIP gave 5.50%. The new experiment changes both reference count and photo variety, so improvements cannot be attributed to count alone. Previous one-positive mAP was reciprocal rank; it should not be interpreted as the same eight-positive retrieval task.

Gallery labels come from public image-search source titles/caption context, with retained original links and search records. All candidate sheets were reviewed for composition and duplicate photographs; ambiguous contexts, obvious renderings and duplicate source-photo crops were excluded before new inference. Exact reference hashes are unique and no reference bytes overlap query bytes. Original downloaded reference pixels are preserved. Source labels are not independently identity-certified.

This remains an exploratory single-match diagnostic using existing ground-truth annotations and sparse four-observation groups, not McByte++ tracks or dense AP3D video clips. The match was inspected previously; labels may contain residual errors. This is not a completed SoccerNet web-gallery benchmark or evidence of production-level accuracy.

[Gallery manifest](../../data/arg_fra_expanded_gallery_v1/gallery.jsonl) · [Gallery folder](../../data/arg_fra_expanded_gallery_v1/gallery/) · [Contact sheets](gallery_sheets/) · [Gallery audit](../../data/arg_fra_expanded_gallery_v1/gallery_audit.json) · [Full results](results.json) · [Independent metric recount](independent_verification.json)

Reproduce scoring: `.venv/bin/python -m experiments.arg_fra_expanded_gallery_v1.run`; recount and report: `.venv/bin/python -m experiments.arg_fra_expanded_gallery_v1.report`. Gallery manifest and checkpoint hashes are saved in each model’s metrics.
