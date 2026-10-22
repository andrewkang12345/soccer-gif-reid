# SoccerNet full named subset v2 — preparation in progress

Experiment: `experiments/soccernet_full_v2`; data: `data/soccernet_full_v2`; reports: `reports/soccernet_full_v2`.

The user approved scaling the soccer pilot to all reliably named public SoccerNet pairs, keeping kit-invariance augmentation without extra visual-quality degradation. The full training run has **not started** and there are **no new full-run performance results** yet.

- Read metadata for all 372 available matches with caption rosters and v3 boxes.
- Join numeric jersey labels only when a unique side-to-roster mapping has at least two independent roster-exclusive number witnesses and no contradictions.
- Found that the caption `teams` list is not ordered home/away. Search/display club names now use explicit `home.name`/`away.name`. The identity join already used the correct side's roster.
- Selected 1,741 candidate players using independent player/game hash partitions; prior pilot and Argentina-France identities cannot enter new validation/test. Ambiguous same-name hashes are excluded.
- Extracted 40,933 crops; segmentation retained 38,640 crops across 1,733 players: 38,308 train, 139 validation, 193 test. These are pre-reference-eligibility counts.
- Independently checked all 41,211 mapped roster joins with zero inconsistencies. Visually reviewed 199 stratified query crops. No clear readable-number contradictions; blur, occlusions and graphics prevent verifying many individual labels. This is weak supervision, not certified identity ground truth.
- Web-image search was rate-limited. Ordinary web search plus image extraction from named public news/photo pages is working. Candidates preserve source URLs, image URLs and page hashes. No Google-only claim.
- Gallery has no eight-image cap. Automatic screening requires name evidence and a dominant person, removes shared-image identity ambiguities and geometrically verified duplicates. Manual sampling found additional duplicate frames and questionable associations; exclusions are saved by candidate ID.
- Two-stage trainer implemented: 790,272-parameter GIF head first, then final two CLIP blocks (26,771,200 trainable parameters), lower backbone learning rate, validation-only selection and early stopping. No added blur/downsample/JPEG/noise degradation.
- GPU gradient smoke passed for both stages; earlier blocks stayed frozen. Resume integration passed: interruption after a fine-tuning checkpoint reproduced uninterrupted parameters within 7.5e-9.
- Inference script supports roster directories, crops/tracklets and a content-hashed reference feature cache. Scores are cosine similarities, not identity probabilities.

Next: complete all-player reference searches/downloads, rerun screening and review, freeze manifest/audit, launch full training, evaluate selected checkpoint and baselines on the same untouched player/game-disjoint test, independently recount metrics and report.

For live artifact counts run `.venv/bin/python -m experiments.soccernet_full_v2.status`. Preparation scripts are resumable from saved search records and downloads; training resumes with `--resume` after launch.
