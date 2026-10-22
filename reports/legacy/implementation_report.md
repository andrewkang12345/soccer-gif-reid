# Implementation report

**2026-09-11 update:** the supplied full MuPNIT-ReID archive has now been extracted, including 30,452 external reference photos for 211 NBA identities. Dataset access is resolved. Training remains unstarted pending folder-based ID parsing, identity-disjoint preparation, reference audit and masks. See the [current resume report](RESUME_2026-09-11.md); the download status below describes the earlier blocked state.

Date: 2026-09-10. Project: `/mnt/data/soccer_gif_reid`.

## Findings in the existing project

- `soccer_identity/identity/body_reid.py` calls the generic `image_embedding` helper. It does not learn broadcast-to-roster body identity.
- `soccer_identity/identity/headshot_matcher.py` compares histogram-style image features from upper crops and reference images. It is not a trained facial or cross-domain ReID model.
- The existing CLIP Market1501, TransReID and OSNet weights target appearance association and do not establish recognition from independent soccer roster photographs.
- `eval_tracking.py` builds `track_to_player` using ground-truth player names and majority votes, then uses that mapping for `identity_acc`. Therefore the reported 89.3% identity accuracy is a GT-assisted track-consistency result, not measured name recognition from roster references. The new pipeline does not reuse that evaluation.
- Original project files were not edited.

## Upstream provenance and checkpoint search

[Official GIF repository](https://github.com/Wojak27/GIF), pinned at `304370c107d9091bea47bb9c5432ac76020c24c3`.

The source of architecture/objective is [`train_clip_projection_layer.ipynb`](https://github.com/Wojak27/GIF/blob/304370c107d9091bea47bb9c5432ac76020c24c3/reid/mupnit_clip_reid/train_clip_projection_layer.ipynb); input/output normalization follows [`clip_reid.py`](https://github.com/Wojak27/GIF/blob/304370c107d9091bea47bb9c5432ac76020c24c3/reid/mupnit_clip_reid/clip_reid.py). The new head has identical layer dimensions and `net.*` state-dict names. Circle Loss uses the same library, with explicit default margin 0.4 and gamma 80.

Checked GitHub files, releases (empty), issues (empty), readme checkpoint links, and the public Baidu checkpoint folder. No trained GIF ReID projection checkpoint was located. The shared checkpoints are named for YOLOX and listed upstream as trained detection weights. This search supports starting from pretrained CLIP; it does not prove that no third-party copy exists anywhere.

The official [paper](https://openaccess.thecvf.com/content/WACV2026/papers/Wojtulewicz_Advancing_Player_Identification_and_Tracking_with_Global_ID_Fusion_GIF_WACV_2026_paper.pdf) describes identity-disjoint train/test splits and a gallery from a different image domain. The implementation preserves these boundaries and creates identity-disjoint validation from training IDs when no explicit validation set is supplied.

## Dataset and execution status

A local search found no MuPNIT archive or image dataset. The official public Baidu password was accepted and the archive listing was retrieved. `MuPNIT-ReID.tar.gz` is 5,913,142,561 bytes; the light archive is 1,691,384,340 bytes. See `baidu_file_listing.json` for the observed filenames/sizes. An initial public-share download request returned `errno: 2`. Adding the public signature succeeded (`errno: 0`) but returned an encrypted string instead of usable links. The standard batch-download path returned links for both archives, but fetching either returned **HTTP 403, error 31064, `file is not authorized`**. No authorization boundary was bypassed. Public Hugging Face dataset search returned no MuPNIT mirror. The actual archive contents and source-domain annotations remain uninspected. The project explicitly requires query/reference source labeling rather than guessing from image quality.

Pretrained OpenAI CLIP ViT-L/14 was downloaded via its official URL and SHA-256 verifier. No unaugmented ReID model was reproduced. No MuPNIT epoch, measured Rank-1/mAP, or trained production checkpoint can be claimed until the data is available.

On the subsequent exact-link retry, Chromium accepted `bw6F`, opened
`MuPNIT/Release_versions`, and selected only `MuPNIT-ReID.tar.gz` (5.51 GiB).
The visible Download button returned the official `baidunetdisk-8.7.0.x86_64.rpm`
installer (211,922,783 bytes), not the dataset. No desktop package was installed into
the OS. The official `bdpan` CLI installer was downloaded from Baidu's CDN, verified
against the SHA-256 published in `baidu-netdisk/bdpan-storage`, and installed only in
`.tools/bdpan/`. Its read-only share-list command returned `请先执行 bdpan login 命令`
(run bdpan login first). Device login stopped at the CLI's first-use terms prompt;
account authorization requires user interaction. `scripts/baidu_login.sh` is the
prepared terminal entry point. Browser logs and screenshots are retained in `reports/`.

## Deliberate changes from the notebook

- Degrade only broadcast/query images; pair every sampled identity with clean references and apply bidirectional cross-domain Circle Loss.
- Keep CLIP frozen initially; optionally train final blocks from the best augmented-head model and promote only on validation improvement.
- Use aspect-preserving letterbox preprocessing consistently, avoiding removal of head/feet by center cropping.
- Use deterministic identity-aware sampling and fixed heavy validation corruption. Cache fixed validation CLIP features only while the encoder is frozen.
- Validate after each epoch; clip gradients **after backward**, save the best completed epoch, retain complete resume state, and use atomic file replacement.
- Separate validation from test; verify identities, paths and content to prevent leakage. Never skip missing gallery positives in metric denominators.
- Package offline-loadable full visual+projection checkpoints and expose reusable crop/tracklet roster inference.

## Known limits

Semantic person masks are generated by a pretrained segmentation model. Unless manually supplied, shirt/shorts regions remain conservative geometry/skin heuristics; preview and improve them with human parsing when feasible. Procedural turf/stands and perspective transforms are domain perturbations, not a complete soccer image distribution or 3D viewpoint synthesis. Exact hashes do not detect every near-duplicate. No cross-domain NBA-to-soccer accuracy, headshot matching reliability, open-set threshold, service load capacity, or deployment latency has been established by implementation tests.

GPU work for validation smoke tests shares the available A10G with existing jobs; those jobs were not stopped. The project uses its own environment and does not replace the host's installed torch/CUDA versions.

## Verification performed

The unit/integration suite passed 26 tests, including exact hand-calculated mAP, missing-positive rejection, identity/content leakage, clean reference preservation, deterministic corruptions and samplers, tiny-image augmentation, frozen/final-block gradient boundaries, projection compatibility, tracklet ranking/rejection, fine-tune promotion, cached/uncached validation agreement, identical epoch-resume model parameters, context-padded segmentation remapping, data mutation, invalid thresholds, and invalid learning-rate rejection.

`real_clip_smoke.json` records actual ViT-L/14 GPU execution: head optimization, frozen visual parameters, self-contained checkpoint save/reload, ranked reference inference, and a backward pass through one final visual block. Temporary smoke weights are deleted; no smoke checkpoint is presented as a trained model. These checks are implementation evidence, not recognition accuracy.

Twelve existing local soccer crops, approximately 35–93 pixels tall, were submitted to person segmentation for augmentation inspection. Eight passed the original mask checks. Context padding recovered three additional predicted masks while preserving and mapping back to the original crop pixels. One remains explicitly rejected; see `data/local_preview_v2/masked_retry.masks.json` and `mask_retry_preview.jpg`. Mask plausibility and visual inspection are not a measured segmentation IoU benchmark. The heavy preview contains deliberately difficult, sometimes ambiguous examples. These crops are not used as MuPNIT training or a soccer evaluation set.
