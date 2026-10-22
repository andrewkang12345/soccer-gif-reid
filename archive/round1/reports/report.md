# Full named SoccerNet v2 result

Selected stage: finetune, epoch 5; validation Rank-1 22.74%, mAP 21.93%.

| Model | Test Rank-1 | Test mAP |
|---|---:|---:|
| soccer_full | 35.29% | 33.63% |
| frozen_clip | 26.80% | 19.05% |
| nba_projection | 11.11% | 11.42% |

Player-cluster bootstrap mAP differences versus baselines: `{"frozen_clip": {"caveat": "Player-cluster resampling does not account for match correlation or weak-label errors.", "mAP_difference": 0.1458369101831461, "player_cluster_bootstrap_95_ci": [0.07177925966368945, 0.22874680005348008]}, "nba_projection": {"caveat": "Player-cluster resampling does not account for match correlation or weak-label errors.", "mAP_difference": 0.22210256863773448, "player_cluster_bootstrap_95_ci": [0.14227909990138363, 0.3071565475371797]}}`

The test players and source games are disjoint from training and validation. Checkpoint selection used validation only. Test metrics were independently recounted from saved embeddings.

Dataset: `{"test": {"games": 26, "players": 33, "queries": 153, "references": 299}, "train": {"games": 281, "players": 219, "queries": 9715, "references": 1703}, "val": {"games": 9, "players": 60, "queries": 343, "references": 383}}`

All screened distinct reference photos were retained, with at least three per eligible player and no maximum cap. Labels remain automatically joined roster/source-context supervision with sampled visual review.

No additional image-quality degradation was used. Query augmentation randomizes kit/background and modest geometry/occlusion; references remain clean.

Inference averages reference features per identity; reported Rank-1/mAP above use image-gallery retrieval, so they measure different aggregation rules.

Argentina-France was development data and is not an independent test for this run. The new test has different players and gallery composition from the small pilot, so cross-run raw percentages are not directly comparable.
