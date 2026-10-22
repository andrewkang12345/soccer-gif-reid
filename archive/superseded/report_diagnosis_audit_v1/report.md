# Training and transfer diagnosis audit

Conclusion: a generalization/transfer problem is supported, especially for CLIP. The stronger claim that missing soccer training pairs is the main cause remains unverified. No soccer-supervised control or training augmentation ablation was run.

New evaluation uses the original v2 NBA-selected best checkpoints, a deterministic first-eight-clips-per-training-identity subset, all 944 training references, and the same clean/appearance-only validation renderer. Training examples are seen data; these scores are diagnostics, not independent validation. Soccer ranking analysis reuses the frozen 168-photo evaluation.

| Model | Train clean R1 / mAP | Train augmented R1 / mAP | NBA validation clean R1 / mAP | Soccer R1 / mAP |
|---|---:|---:|---:|---:|
| clip | 45.03% / 42.94% | 43.32% / 40.47% | 24.15% / 23.60% | 12.84% / 13.21% |
| ap3d | 27.70% / 27.76% | 31.44% / 29.89% | 11.46% / 12.49% | 6.42% / 8.14% |

Training diagnostic: 935 clips across 118 identities; NBA validation 323 clips across 42; soccer 109 across 20 query/21 gallery identities. Different candidate counts and seen identities limit direct numeric comparability.

CLIP soccer-selected-run epoch 1 → 9: training loss 33.98 → 5.21; NBA clean mAP 15.81% → 21.07%; soccer mAP 13.20% → 9.75%. AP3D soccer mAP peaked at epoch 4 (9.00%) and ended epoch 12 at 8.66%; evidence of a decline is much weaker for AP3D.

CLIP smaller-half soccer queries: 3/56 correct (5.36%); larger half: 11/53 (20.75%). Size is mean original crop height, median 58.25 pixels. This is an association, confounded by identity, pose, occlusion and annotation differences; it does not measure visible face information. AP3D: 3/56 versus 4/53.

Oracle known-team filtering raises CLIP R1 only from 12.84% to 13.76%, AP3D from 6.42% to 10.09%. Wrong top predictions are same-team in 50/95 CLIP errors and 41/102 AP3D errors. Therefore a claim that errors are predominantly teammate confusion is not supported.

Augmentation at evaluation reduces CLIP training-identity mAP from 42.94% to 40.47%; AP3D increases from 27.76% to 29.89%. This does not establish that training augmentation harms transfer: these models were trained with it. Only a matched training ablation could establish its contribution.

Verified: models generalize poorly beyond training identities, CLIP NBA gains do not transfer to this match, and smaller crops associate with lower CLIP accuracy. Unverified: missing soccer pairs as the dominant cause, insufficient facial detail as the exact mechanism, and augmentation as the cause. Reference-label errors, preprocessing/task mismatch, and optimization remain possible contributors.

Next causal experiment: compare matched NBA-only and NBA+soccer-pair training with the same kit augmentation and equal update budget, using different soccer players/matches for evaluation. Argentina–France has already been used for checkpoint selection, so it cannot be the independent evaluation for that claim.

Reproduce: `.venv/bin/python -m experiments.diagnosis_audit_v1.audit`. Raw results, selected clip IDs and histories are in `results.json`.
