# Large files nothing in the current pipeline reads

Nothing here has been deleted. Each line says what it is and what would be lost.
Sizes from `du`; total about 130 GB.

| path | size | what it is | risk if removed |
|---|---:|---|---|
| `data/MuPNIT_60fps.tar.gz` | 82 G | the archive of `data/MuPNIT_60fps`, which is already extracted | none beyond re-download; the extracted copy remains |
| `data/mupnit_dataset.zip` | 13 G | original MuPNIT download; `data/MuPNIT-ReID` is the extracted form | re-download only |
| `data/trash/mupnit_dataset.repaired-zip64.zip` | 13 G | a repaired copy of the same zip, in a directory named trash | none |
| `data/MuPnit_YoloX_weights` | 12 G | YoloX detector weights shipped with MuPNIT; this pipeline uses yolo11m-seg from sports_reID | re-download only |
| `data/MuPNIT-ReID.tar.gz` | 5.6 G | archive of `data/MuPNIT-ReID`, already extracted | none beyond re-download |
| `data/MuPNIT-ReID-light.tar.gz` | 1.6 G | the light variant, never extracted or read | re-download only |
| `data/baidunetdisk-8.7.0.x86_64.rpm` | 203 M | installer used once to fetch the MuPNIT dataset | none |

`data/MuPNIT_60fps` (83 G) and `data/MuPNIT-ReID` (6.6 G) are the extracted NBA data. The only
thing the current pipeline needs from MuPNIT is the trained checkpoint
`runs/ap3d_no_quality_v2/clip/best.pt`, which is scored as a baseline every run and is 1.2 GB.
Removing the extracted data would end the ability to retrain that baseline from scratch.
