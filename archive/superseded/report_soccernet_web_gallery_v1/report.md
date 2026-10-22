# Soccer evaluation status — 2026-09-12

**The requested SoccerNet → named web-photo test is prepared but not completed. No SoccerNet web-gallery accuracy is available.**

## Completed preparation

- Frozen query selection from the local official SoccerNet-GSR test archive: 9 clips from games 7, 8 and 11.
- 1,144 non-overlapping four-frame samples, 4,576 original-quality full-bbox crops; 143 anonymous clip/team/jersey keys. These are not 143 verified real people.
- Loaded our NBA-validation-selected CLIP-Proj epoch-24 and AP3D epoch-22 checkpoints and extracted features for every query. Both models used the same source observations.
- Verified the source archive SHA256, saved PNG hashes, frame spacing, distinct observations within each query, finite model features and feature/manifest alignment. No added degradation, OCR filter, training, checkpoint selection or threshold tuning.

## Missing input

The local jersey-recognition archive has track-number labels without names or match identifiers. GSR has anonymous game IDs, clip-relative team sides and jersey numbers, without match dates, club names or player names. The original tracking archive was inspected by HTTP range requests and its `gameinfo.ini` has the same anonymous labels. The authors explain that these matches differ from the main SoccerNet video collection in [this mapping discussion](https://github.com/SoccerNet/sn-tracking/issues/7).

The [raw tracking videos](https://huggingface.co/datasets/SoccerNet/SoccerNet-Tracking-RAW-Video) are gated. The standard authenticated Hugging Face metadata call for the video archive returned `GatedRepoError` (HTTP 403). No access request or agreement was submitted. Their README was readable, but that did not grant access to the video archive.

A verified match/roster mapping is needed before attaching named external photos to query identities. Left/right labels must be mapped per clip, not assumed to represent a fixed club. No player names were guessed from appearance or model predictions. A mapping template is in `data/soccernet_web_gallery_v1/identity_mapping_required.csv`.

Target gallery: eight distinct external photos per named player, matching the eight individual references per identity in the NBA test. No Google/web gallery for these anonymous SoccerNet players was created. A Google HTTP image-search probe returned a JavaScript redirect with no usable image results. Once names are known, collection and source/duplicate/content review remain necessary before scoring.

## Separate existing Argentina–France diagnostic

This provides an immediate test of our trained models on real soccer, but **does not satisfy the requested SoccerNet/eight-photo test**. The diagnostic uses the previously prepared 109 sequences of four adjacent sparse annotation samples, 20 query identities and 21 existing roster portraits, one per reference identity. The 21-person gallery is shared by all three models.

| Model | Correct / 109 | Rank-1 | mAP |
|---|---:|---:|---:|
| Frozen CLIP backbone | 6 | 5.50% | 16.08% |
| Our trained CLIP-Proj | 14 | 12.84% | 26.13% |
| Our trained AP3D | 12 | 11.01% | 26.06% |

All three use checkpoint-native aspect-preserving letterboxing. The backbone baseline comes from the frozen visual encoder in the trained CLIP checkpoint and omits its projection head. Its result is the fair preprocessing-matched comparison for projection training; the earlier 4.59% vanilla baseline used official center cropping. AP3D receives the same four observations, but these sparse annotation samples differ from dense training clips.

There is one reference per identity, so mAP equals reciprocal rank here. Prediction dumps were independently recounted; a separate hand-computable multi-positive example verifies AP arithmetic for the planned eight-reference setting. No result demonstrates production readiness. This is a reused, imperfectly annotated clip, with inherited portrait labels and GT boxes/identity grouping rather than McByte++ tracks.

## Artifacts

- [Status and checks](status.json)
- [SoccerNet preparation counts](soccernet_preparation.json)
- [Evaluation workflow](../../experiments/soccernet_web_gallery_v1/README.md)
- Query manifest: `data/soccernet_web_gallery_v1/queries.jsonl`
- Cached query embeddings: `reports/soccernet_web_gallery_v1/soccernet_query_features/{clip,ap3d}/query_features.pt`
