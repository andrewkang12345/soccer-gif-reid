# SoccerNet named-player training — 2026-09-13

Completed isolated `soccernet_named_v1` pilot. 38/16/24 real-player-disjoint train/validation/test identities; source matches also disjoint. Kit-invariance augmentation retained with 16 cached variants per training crop; no added quality degradation. Fresh CLIP-Proj head trained to early stop, selected epoch 4. NBA-head fine-tuning retained epoch zero.

Held-out 134-crop/24-player test: frozen CLIP 17.91% R1 / 13.75% mAP; NBA head 11.94% / 14.52%; fresh soccer head 13.43% / 18.84%; fine-tuned head matches NBA because epoch zero selected. Independent metric recount passed. Bootstrap gain versus NBA includes zero. These scores do not establish production performance.

[Full report](soccernet_named_v1/report.md) · [Protocol](soccernet_named_v1/protocol.md).
