# soccer_gif_reid — layout

Match a degraded broadcast player crop against a roster of clean reference photographs.
`reports/EXPERIMENT_STATUS.md` is the current status document; its §14 contains the historical run log.
The October 20 audit found evaluation identities in older auxiliary pretraining. A fresh v8 run
skips that stage; older scores must carry the leakage caveat.

## Live

| path | what |
|---|---|
| `experiments/soccernet_full_v2/` | the current pipeline: collection, screening, splits, training, scoring, visualisation |
| `data/soccernet_full_v2/` | frozen manifest, gallery, query crops, audits, self-supervised crops |
| `runs/soccernet_full_v8/` | clean paired-data-only run; inspect status.json or status.py |
| `runs/soccernet_full_v7/` | latest completed model; auxiliary identity leakage found, historical comparison only |
| `reports/soccernet_full_v2/` | protocol, screening audits, review sheets, result, error sheets |
| `reports/EXPERIMENT_STATUS.md` | status and run log — read this first |
| `src/soccer_gif_reid/` | model, augmentation, metrics, shared helpers |
| `weights/` | CLIP ViT-L/14 backbone |

## Kept because something still points at it

| path | why |
|---|---|
| `runs/ap3d_no_quality_v2/clip/best.pt` | the NBA-trained projection, scored as a baseline every run |
| `data/arg_fra_expanded_gallery_v1/` | Argentina-France development clip and its gallery |
| `data/soccernet_named_v1/` | SoccerNet ReID archives and the 38-player pilot the splits exclude |
| `experiments/soccernet_named_v1/`, `experiments/ap3d_no_quality_v2/` | imported by the current pipeline |
| `archive/round1/` | R1's run, data snapshot and report |

## Archived

`archive/superseded/` holds experiments, runs and reports that nothing imports any more:
the AP3D comparison, the Argentina-France validation runs, the web-gallery and released-GIF
probes, the diagnosis audit, and the 38-player pilot run. `reports/legacy/` holds one-off
install logs, browser probes and superseded daily logs.

## Data that is large and no longer read

See `archive/DELETION_CANDIDATES.md`. Nothing has been deleted.
