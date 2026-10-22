# soccernet_full_v8: completed soccer ReID run

Verified 2026-10-21T01:13:10.969354+00:00. Selected finetune epoch 3 using validation only.

Auxiliary pretraining epochs: 0. Manifest: `4f9de598127cd103783728c1127ab00db84ec23193c85ce6284f04ab6cca1ab6`.

| Photo retrieval within match | Rank-1 | mAP |
|---|---:|---:|
| Frozen CLIP | 14.85% | 19.12% |
| NBA projection | 14.99% | 23.18% |
| Soccer projection | 23.07% | 28.23% |

| Photo + known roster number | Rank-1 | Top-3 | MRR |
|---|---:|---:|---:|
| Individual query crop | 43.44% | 64.43% | 57.97% |
| Player-by-match pooled observations | 55.74% | 79.23% | 69.98% |

Photo weights chosen on validation: `{"per_crop": 0.5, "per_tracklet": 0.5}`.

Dataset:

| Split | Players | Queries | References | Matches |
|---|---:|---:|---:|---:|
| train | 265 | 9140 | 2123 | 286 |
| val | 115 | 1043 | 324 | 12 |
| test | 182 | 1448 | 818 | 12 |

The photo table ranks individual reference photographs. Fusion ranks one prototype per identity, so MRR and image-gallery mAP should be interpreted separately.

Supervised identities, normalized names, matches and hashes are disjoint. This run starts from pretrained CLIP and uses no earlier soccer checkpoint. No added blur/downsampling/JPEG/noise degradation was enabled.

The older v7 auxiliary stage contained evaluation identities, so its scores are historical diagnostics. This test set has been reused during development; these results are not a newly untouched final test. Ground-truth player-by-match pooling assumes perfect grouping and is not McByte++ tracking performance. Candidate rosters include reference-covered labelled players rather than every active squad member.

There is no validated production commitment threshold. The earlier Argentina-France demo failed to transfer; a completed SoccerNet score does not establish production readiness.

Checkpoint: `runs/soccernet_full_v8/best.pt`. Detailed audit: `verification_soccernet_full_v8.json`. Training history: `runs/soccernet_full_v8/history.jsonl`.
