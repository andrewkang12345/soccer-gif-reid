> Historical v6 photo-retrieval report. The October 20 audit found held-out identities in auxiliary pretraining. See [current status](../EXPERIMENT_STATUS.md) for the clean v8 restart and current verification.

# Full named SoccerNet v2 result

Selected stage: finetune, epoch 1; validation Rank-1 14.57%, mAP 16.86%.

Per-match retrieval is the deployment setting: the gallery is the roster of the one match the
crop came from, so every candidate wears one of two kits and the answer is bounded to about
twenty people. Whole-split retrieval ranks against every test identity at once.

| Model | Per-match Rank-1 | Per-match mAP | Whole-split Rank-1 | Whole-split mAP |
|---|---:|---:|---:|---:|
| soccer_full | 25.62% | 28.79% | 11.40% | 10.86% |
| frozen_clip | 14.85% | 19.12% | 3.87% | 4.44% |
| nba_projection | 14.99% | 23.18% | 2.35% | 4.44% |

Per match, for the selected model:

| match | players | queries | Rank-1 | mAP |
|---|---:|---:|---:|---:|
| 2015-09-19 - 19-30 Manchester City 1 - 2 West Ham | 20 | 166 | 16.27% | 17.80% |
| 2016-10-01 - 14-30 Swansea 1 - 2 Liverpool | 16 | 183 | 27.87% | 28.88% |
| 2016-11-06 - 19-30 Leicester 1 - 2 West Brom | 15 | 119 | 25.21% | 30.89% |
| 2014-11-04 - 20-00 Zenit Petersburg 1 - 2 Bayer Leve | 15 | 120 | 33.33% | 39.53% |
| 2015-09-15 - 21-45 Galatasaray 0 - 2 Atl. Madrid | 13 | 54 | 29.63% | 37.53% |
| 2015-09-15 - 21-45 PSV 2 - 1 Manchester United | 12 | 56 | 14.29% | 21.22% |
| 2015-11-03 - 22-45 Shakhtar Donetsk 4 - 0 Malmo FF | 17 | 107 | 37.38% | 30.48% |
| 2015-11-25 - 22-45 B. Monchengladbach 4 - 2 Sevilla | 13 | 134 | 14.93% | 24.37% |
| 2016-12-20 - 22-00 Dortmund 1 - 1 FC Augsburg | 19 | 199 | 19.60% | 22.01% |
| 2015-04-19 - 21-45 Inter 0 - 0 AC Milan | 14 | 53 | 26.42% | 35.76% |
| 2016-11-20 - 17-00 Atalanta 2 - 1 AS Roma | 16 | 125 | 41.60% | 37.68% |
| 2017-04-15 - 21-45 Barcelona 3 - 2 Real Sociedad | 13 | 132 | 25.76% | 32.56% |

Player-cluster bootstrap mAP differences versus baselines: `{"frozen_clip": {"caveat": "Player-cluster resampling does not account for match correlation or weak-label errors.", "mAP_difference": 0.06416480765710293, "player_cluster_bootstrap_95_ci": [0.046544735068830445, 0.08400327276924645]}, "nba_projection": {"caveat": "Player-cluster resampling does not account for match correlation or weak-label errors.", "mAP_difference": 0.06420038268511556, "player_cluster_bootstrap_95_ci": [0.043319058517500036, 0.0869200282788272]}}`

The test players and source games are disjoint from training and validation. Checkpoint selection used validation only. Test metrics were independently recounted from saved embeddings.

Dataset: `{"test": {"games": 12, "players": 182, "queries": 1448, "references": 818}, "train": {"games": 286, "players": 265, "queries": 9140, "references": 2123}, "val": {"games": 12, "players": 115, "queries": 1043, "references": 324}}`

All screened distinct reference photos were retained, with at least three per eligible player and no maximum cap. Labels remain automatically joined roster/source-context supervision with sampled visual review.

No additional image-quality degradation was used. Query augmentation randomizes kit/background and modest geometry/occlusion; references remain clean.

Inference averages reference features per identity; reported Rank-1/mAP above use image-gallery retrieval, so they measure different aggregation rules.

Argentina-France was development data and is not an independent test for this run. The new test has different players and gallery composition from the small pilot, so cross-run raw percentages are not directly comparable.
