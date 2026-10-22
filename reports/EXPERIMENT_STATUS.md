# Soccer player re-identification — full experiment status

_Updated 2026-10-22. Current model: `runs/soccernet_full_v8/best.pt` (completed and verified
2026-10-21, paired-data only). Previous run: `runs/soccernet_full_v7/` (historical, auxiliary
identity leakage found). Frozen supervised manifest: `4f9de598127cd103…`._

## 1. Current state

Full training and later experiments finished on September 25. On October 20, the saved results
were independently recounted and the data hashes checked. A deeper audit then found that the
auxiliary self-supervised stage contained evaluation identities in other matches. **The earlier
runs with that stage are not strictly player-disjoint benchmarks.** Their supervised manifests
are disjoint; their complete training inputs are not.

A fresh run, **v8**, retrained from pretrained CLIP ViT-L/14 with a fresh projection and number
adapter. It uses the same frozen paired dataset, skips auxiliary pretraining, trains the head
first, and then fine-tunes the last two CLIP blocks. It retains the existing kit/background
augmentation and adds no blur, downsampling, JPEG or noise degradation. It completed and passed
independent verification on 2026-10-21 (§4, R9 in §14).

**Without the leaked auxiliary stage, v8 matches v7.** Fused photo+number test scores are
Rank-1 43.44% per crop and 55.74% per ground-truth player-by-match pool, against v7's 41.09% and
54.10%. Photo-only per-match Rank-1 is 23.07% against 25.35%. No paired difference, fused or
photo-only, is distinguishable in a player-clustered bootstrap. The auxiliary stage, leak included,
was not what produced v7's scores. V8 is the current model; v7 remains a historical diagnostic.

## 2. The leakage audit

The old guard excluded validation/test **matches** from auxiliary pretraining. That did not
exclude the same players appearing in different training matches. Of the 40,000 auxiliary groups
used by v7, 8,204 could be resolved through numeric jersey labels, original box UAI and accepted
team/roster joins. **1,508 groups contained held-out identities: 48 validation players and
130 test players.** Anonymous groups remain unresolved, so these counts are lower bounds.

Evidence: `reports/soccernet_full_v2/ssl_identity_audit.json`. The audit uses original UAI because
the archive's action index is global and cannot be joined directly to a match-local frame name.

Auxiliary pretraining is now disabled by default. If explicitly enabled in a new run, the trainer
excludes entire matches whose rosters contain any held-out player ID or normalized name, and
excludes matches without roster metadata. That conservative filter retains 5,817 of 68,985
available auxiliary groups over 28 matches and removes every identified leaked group. V8 uses
none of them, so its supervised manifest is its only soccer training-image source. CLIP's original
web pretraining overlap remains unknown.

## 3. Frozen paired dataset

| Split | Players | Query crops | Reference photos | Matches |
|---|---:|---:|---:|---:|
| Train | 265 | 9,140 | 2,123 | 286 |
| Validation | 115 | 1,043 | 324 | 12 |
| Test | 182 | 1,448 | 818 | 12 |

Source metadata covers 372 matches and 1,741 candidate named identities. This is the usable
named/photo-paired subset, not every anonymous SoccerNet player. Training requires at least
three distinct references; evaluation allows one to avoid selecting only well-photographed
players. References are uncapped: training has 3–56 photos/player, validation 1–21 and test 1–22.
They come from public named pages and Wikidata/Commons entity links, not Google alone.

References are padded dominant-person crops, written losslessly; source files, source hashes and
crop boxes are recorded. Stored references remain clean; training also applies the existing
reference-kit augmentation. Numeric jersey/roster joins and source-context references remain
weak supervision, with sampled manual review rather than complete identity certification.

Supervised player ID, normalized name, source match and image hashes are disjoint across splits.
On October 20, all **14,896 stored query/reference images**, all **3,265 reference source files**,
and the query-mask file paths were checked. These checks cover the frozen supervised manifest;
they do not retroactively make the old auxiliary stage player-disjoint.

## 4. Results: clean v8 (current) and historical v7

V7 scores are retained for diagnosis, **with the auxiliary identity-leak caveat**; v8 has no
auxiliary stage. Neither is a new untouched test: this test set was reused across recorded
experiments.

Individual-photo retrieval, restricting each query's candidate gallery to the reference-covered
players observed in its labelled match:

| Model | Rank-1 | mAP |
|---|---:|---:|
| Frozen CLIP ViT-L/14 | 14.85% | 19.12% |
| NBA-trained MuPNIT projection | 14.99% | 23.18% |
| Soccer v7, photographs only (leaky) | 25.35% | 29.50% |
| **Soccer v8, photographs only** | **23.07%** | **28.23%** |

Photo/number prototype fusion, each run at its validation-selected photo weight (v7 0.4, v8 0.5):

| Evaluation unit | v7 Rank-1 / Top-3 / MRR | **v8 Rank-1 / Top-3 / MRR** |
|---|---:|---:|
| Individual query crop (1,448) | 41.09 / 64.02 / 56.59 | **43.44 / 64.43 / 57.97** |
| Ground-truth player-by-match pool (183) | 54.10 / 81.97 / 69.82 | **55.74 / 79.23 / 69.98** |

V8 minus v7, paired and clustered by player (5,000 draws), **none distinguishable**: per crop
Rank-1 +2.35 [-0.68, +5.27], MRR +1.38 [-0.95, +3.67]; per pool Rank-1 +1.64 [-4.40, +7.65],
Top-3 -2.73 [-8.11, +2.20], MRR +0.16 [-3.63, +3.79]; photo-only per-match Rank-1 -2.28
[-5.52, +0.92], mAP -1.27 [-3.10, +0.45]. Evidence:
`reports/soccernet_full_v2/fused_comparison_soccernet_full_v8_vs_soccernet_full_v7.json`,
`run_comparison_v8_vs_v7.json`, `results_soccernet_full_v8.md`, `verification_soccernet_full_v8.json`.

Pooling uses known identity labels to group observations; these are not McByte++ tracklets and
this does not measure tracker switches or time to identification. Candidate rosters contain
labelled, reference-covered players, not every possible squad member. Fusion ranks one prototype
per identity, while the first table ranks individual reference images; the metrics differ.

V8 selected fine-tuning epoch 3 on fused validation MRR 66.76%; v7 selected fine-tuning epoch 5
at 65.46%. V7's improvement over v6 was not statistically distinguishable in the recorded player
bootstrap, even before the leakage caveat. Whole-split photo retrieval, across all 182 test
identities, was Rank-1 7.80%, mAP 8.96% for v8 and 10.29%, 10.34% for v7 (frozen CLIP 3.87%,
4.44%). Number reading within the match roster: v8 33.8%, v7 31.4%, chance 7.2%.
Independent float64 recounts can reorder nearly tied images relative to float32 metrics.

## 5. Argentina–France remains poor

The v7 streaming demonstration named 18 of 21 tracks and got **5 correct: 27.8% committed-name
precision**, 85.7% coverage. Its validation-calibrated margin did not transfer. It assumes annotated
tracks, and Argentina–France identities are development/training identities, so it is not an
independent test. The clip's median crop area is about 1,500 pixels versus about 10,399 in the
SoccerNet test; the number channel is unreliable at that size. No resolution degradation has
been reintroduced. These results do not meet the production precision objective. The
Argentina–France demonstration has not been rerun with v8.

## 6. Clean v8 training

Fresh pretrained CLIP initialization; 500 identity-balanced batches/epoch, 16 identities/batch,
seed 42; head maximum 30 epochs and final-two-block fine-tuning maximum 15. Auxiliary epochs and
auxiliary persons are both zero. The number loss weight is 2.0, flat-kit probability 0.2, unknown
number target uniform, and selection uses fused validation MRR. The same test remains a reused
development benchmark, not a newly untouched final test.

The early-stopping counter was also corrected: patience now counts non-improving epochs on the
selected fused metric independently of whether photo-only mAP improved. Older runs' checkpoints
and histories are preserved.

Outcome: the head ran 29 epochs (last improvement epoch 21) and fine-tuning 11 (selected epoch 3,
validation fused MRR 66.76%, Rank-1 52.35% at photo weight 0.5), about 4.5 hours at ~400 s/epoch.
The photo-selected checkpoint `best_photo.pt` is head epoch 6 (validation fused MRR 63.91%).

## 7. Inference and verification

The shared loader now supports both native-quality checkpoint formats and constructs the number
adapter when its weights are present. Previously the general inference script could not open the
newer v3-format checkpoints. `infer.py` now optionally fuses known roster numbers with photo
centroids using `--roster-numbers` and an explicit `--photo-weight` chosen on validation.

V8 passed the same completion verification: all 14,896 stored images and 3,265 source photos
rehashed, supervised split separation and soccer training-identity disjointness confirmed with the
auxiliary stage disabled, cached features identical to training features (max difference 0.0),
and an independent recount of every reported score.

Verified on the completed v7 checkpoint: cached/uncached rankings agree exactly; independently
computed fused scores differ by at most 1.5e-8; fresh query embeddings agree with saved embeddings
at minimum cosine 0.9999989; the older photo-only v5 checkpoint still loads. These are software
correctness checks, not additional accuracy measurements. Similarities are not calibrated
identity probabilities, and there is no validated production auto-assignment cutoff.

Artifacts: `verification_soccernet_full_v8.json`, `verification_soccernet_full_v7.json`, `current_inference_check.json`,
`ssl_identity_audit.json`, and `ssl_guard_check.json` in `reports/soccernet_full_v2/`.

## 8. Commands

```bash
cd /mnt/data/soccer_gif_reid
.venv/bin/python -m experiments.soccernet_full_v2.status
# Reproduce the clean v8 run from scratch, or resume an interrupted one:
bash experiments/soccernet_full_v2/run_identity_disjoint.sh --resume
# Paired, player-clustered comparisons of two completed runs (CPU, cached features):
.venv/bin/python -m experiments.soccernet_full_v2.compare_fused runs/soccernet_full_v8 runs/soccernet_full_v7
.venv/bin/python -m experiments.soccernet_full_v2.compare_runs runs/soccernet_full_v8 runs/soccernet_full_v7 \
  --out reports/soccernet_full_v2/run_comparison_v8_vs_v7.json
```

Photo-only roster inference with the selected v8 checkpoint:

```bash
.venv/bin/python -m experiments.soccernet_full_v2.infer \
  --checkpoint runs/soccernet_full_v8/best.pt \
  --roster /path/to/roster --query /path/to/crop_or_tracklet \
  --cache /path/to/roster_features.pt --top-k 5
```

Use `roster/<player_name>/*.jpg`; query may be one crop or a folder of genuine observations.
Optional roster-number JSON maps exact folder names to integers 0–99; duplicate numbers across
players are allowed. Supply `--roster-numbers /path/to/numbers.json --photo-weight VALUE`, using
that run's validation-selected alpha (0.5 for v8). Roster photos are detector-cropped by default; use
`--no-roster-crop` for already prepared person crops.

## 9. Next result to assess

V8 is scored and verified; it is the clean reference point for further changes, and the leaky v7
scores are historical diagnostics only. Every v8 gain over v7 sits inside the noise of 182 test
players, so a future change needs a paired `compare_fused` interval, not a point difference.
Still open: rerunning the Argentina–France demonstration with v8, whose small crops were v7's
failure mode. A final production claim requires new untouched matches, genuine tracker output and
commitment calibration on crop qualities representative of the target broadcast. Do not silently
re-enable resolution degradation.

## 10–13. Historical context

Earlier collection limits, protocol changes, results and proposed next steps are preserved in the
chronological log below. Older entries labelled “running” or “in flight” describe their original
dates and are superseded by the current state above.

## 14. Run log

Append a new entry at the **bottom**, oldest first. One entry per dataset freeze, whether or not
it produced a result — a discarded run is as informative as a kept one, and an undocumented one
invites repeating it. Copy the template, fill every field, and write "none" rather than deleting
a heading.

<details>
<summary>Template</summary>

```markdown
### R<n> — <status> — frozen <UTC timestamp>

- **Manifest** `<final_audit.manifest_sha256, first 16>`
- **Splits** train <p>/<q>/<r>, val <p>/<q>/<r>, test <p>/<q>/<r> (players/queries/references)
- **Evaluation** <per-match | whole-split>, test players per match <list>
- **Changed from previous** <one or two sentences; what and why>
- **Training** <stages, epochs run, selected epoch, validation mAP at selection>
- **Result** <per-match and whole-split Rank-1/mAP, plus both baselines; or "none — discarded">
- **Visualisation** `<path>`
- **Findings** <what the numbers and the error sheets showed that the previous run did not>
- **Disposition** <kept as current | superseded by R<n+1> | discarded, reason>
- **Artifacts** `<where the run and its data snapshot live>`
```

</details>

### R1 — completed — frozen 2026-09-15T04:57Z

- **Manifest** `5abe05e534027fb5`
- **Splits** train 219/9715/1703, val 60/343/383, test 33/153/299
- **Evaluation** whole-split only; test was the original independent player/game hash partition, so its 33 players were scattered over 26 matches (one to five each)
- **Changed from previous** First freeze that was possible at all. The inherited preparation had
  three usable validation players because search queries were batched four at a time and each
  saved record held the merged result set. Added entity-linked references, changed the gallery to
  the dominant-person crop, widened name evidence, and re-cut validation games-first.
- **Training** head 16 epochs (best 8), fine-tune 15 (best 5, selected); validation mAP 21.93% at selection
- **Result** whole-split Rank-1 **35.29%**, mAP **33.63%**; frozen CLIP 26.80% / 19.05%;
  NBA-trained projection 11.11% / 11.42%. Bootstrap mAP difference vs frozen CLIP +14.58 pts
  (95% CI 7.18–22.87), vs NBA +22.21 pts (14.23–30.72). Recount matched to 1e-6.
- **Visualisation** `reports/soccernet_full_v2/test_visualisation/` (generated afterwards, whole-split gallery)
- **Findings** The NBA-trained projection scores *below* frozen CLIP — a MuPNIT head does not just
  fail to transfer, it destroys usable CLIP structure. The head saturated by epoch 8 while training
  loss fell fivefold: 219 identities being memorised. Error sheets showed retrieval
  collapsing to kit colour — every Roma-kit query returned a Roma-kit reference at 0.80–0.89
  similarity — because queries are kit-randomised but references never are.
- **Disposition** superseded by R2; kept as the only run with a published result so far
- **Artifacts** `archive/round1/` (run, data snapshot, report, training log)

### R2 — discarded — frozen 2026-09-15, manifest not retained

- **Manifest** not retained
- **Splits** train 312/12,770/2,287, val 61/375/447, test 34/157/306
- **Evaluation** whole-split; test still the original hash partition
- **Changed from previous** Free page re-mining. 7724 saved pages were re-read against the
  whole roster instead of only the player whose search record listed them — 4,115 were already
  cached holding ~96,000 captioned images, each previously mined against one name. Page candidates
  10501 → 22056, downloaded images 8,903 → 16,264, at zero cost. Also fixed a screen that
  re-tested caption-mined images against the *page title*, rejecting 6,311 of them, and stopped
  one-word roster names ("Bruno") matching inside longer names ("Bruno Genesio") in captions.
- **Training** stopped at head epoch 5
- **Result** none — discarded
- **Visualisation** none
- **Findings** The new image-caption evidence class is the weakest yet: ~5% wrong identity in 72
  reviewed, against ~1% for the older classes, mostly article headlines used as alt text putting
  the wrong person under the right name.
- **Disposition** discarded — re-cutting the test split games-first moved training players into
  test, so the partly-trained model was contaminated and the run had to restart
- **Artifacts** none kept; its dataset improvements carried into R3

### R3 — completed — frozen 2026-09-15T18:46Z

- **Manifest** `0cfca59d62ed6118`
- **Splits** train 282/10,302/2,145, val 64/392/448, test 61/375/447
- **Evaluation** per-match (gallery = that match's roster), whole-split reported alongside; test
  players per match 21, 20, 20, 17
- **Changed from previous** Test cut games-first so it can be scored the way the model is used.
- **Training** head 10 epochs (best 2), fine-tune 13 (best 5, selected); validation mAP 18.70%
- **Result**

  | model | per-match Rank-1 | per-match mAP | whole-split Rank-1 | whole-split mAP |
  |---|---:|---:|---:|---:|
  | soccer_full | 22.40% | 23.72% | 12.53% | 12.92% |
  | frozen CLIP | 14.67% | 16.91% | 6.67% | 8.19% |
  | NBA projection | 13.87% | 19.10% | 4.80% | 6.90% |

  Per match: Shakhtar–Malmo 34.69% / 28.73%, Man City–West Ham 20.44% / 22.32%,
  Leverkusen–Atlético 18.92% / 19.26%, Frankfurt–Dortmund 14.56% / 22.41%.
  Bootstrap whole-split mAP difference vs frozen CLIP +4.73 pts
  (95% CI 1.27–8.06), vs NBA +6.02 pts (3.58–8.72). Both exclude zero.
- **Visualisation** `reports/soccernet_full_v2/test_visualisation/` — 25 sheets, ranked against the
  match roster; 120 of 291 misses still put the true identity in the top five
- **Findings**
  - **R1's strongest claim does not survive the better protocol.** R1 reported the NBA-trained
    projection far below frozen CLIP (6.90% vs 8.19% mAP there) and I described it as destroying
    usable CLIP structure. Per match the ordering reverses on mAP: NBA 19.10% against frozen
    CLIP 16.91%, with Rank-1 near-tied. Whole-split retrieval rewards kit colour, which
    the NBA head has no soccer prior to exploit; removing that reward removes most of its
    apparent deficit. The trained model still leads both on every measure.
  - The margin over frozen CLIP narrowed from +14.58 to +4.73 mAP points. Much of R1's headline
    gap was the colour shortcut, not identity.
  - Per-match scores vary three-fold across four matches (14.6% to 34.7% Rank-1), so
    four matches is too few to rank methods confidently.
  - 120 of 291 misses keep the true player in the top five of about twenty, so tracklet
    aggregation and a roster prior have real headroom above single-crop Rank-1.
- **Disposition** current
- **Artifacts** `runs/soccernet_full_v2/`, `reports/soccernet_full_v2/training_r3.log`, `report.md`

### R4 — completed — frozen 2026-09-15T20:16Z — all planned enhancements

Merges everything previously logged as R4 through R8, because they attack one problem: with this
training data the model can succeed by reading kit colour and shirt numbers rather than
recognising people, which the R3 error sheets showed it doing.

- **Manifest** `985935f908e6f463`
- **Splits** train 285/10376/2162, val 64/392/464, test 61/375/459.
  Test identities and queries are **identical to R3** (61 players, 375 queries); only the test gallery grew by
  12 photographs, so the two runs are directly comparable.
- **Evaluation** per-match, unchanged from R3
- **Changed from previous**
  - *Reference kit augmentation* (`augment_v2.ReferenceAugment`). The reference is recoloured too,
    so kit colour cannot be the cue shared inside a positive pair. The reference keeps its own
    background, pose and photographic quality — only the clothing is repainted.
  - *Shirt-number erasure* (`flat` recolour). The old recolour preserved the digits' luminance
    pattern; filling the clothing flat removes them. Applied with probability to queries,
    references and self-supervised views, so both regimes are still seen.
  - *Same-kit negatives* (`MatchPairs`). Three quarters of batches take their core from one match,
    so the players the loss must separate wear one of two kits. Matches rarely hold sixteen
    eligible players, so the match supplies a core and the rest is filled at random.
  - *Self-supervised stage on unlabelled SoccerNet* (`ssl_pairs.py`, stage 0). 72,127 persons with
    172,008 crops, positives being the same annotated person across linked camera views of one
    action. From annotation, not a tracker, so no identity switch can poison a pair; and two
    thirds of these crops have no readable number, making them invisible to the labelled pipeline.
    All 10 held-out matches are excluded and the trainer asserts it.
  - *Coverage* (`collect_squads.py`). 69 club domains recovered from data already held, 2208 squad
    paths crawled, 3595 new candidates — and **+3 eligible players**. Squad pages list current
    squads against 2014-2017 matches. Recorded as a dead end.
  - *Reference aggregation* (`aggregate.py`). Validation features are now saved, so mean prototype,
    best-single-reference and k prototypes can be compared and the rule chosen on validation.
- **Training** three stages (ssl 8 epochs, head 11, fine-tune 10; selected finetune epoch 2, validation mAP 18.99%)
  three stages: self-supervised (8 epochs, head + last two CLIP blocks), head on frozen
  backbone, then fine-tune. The self-supervised stage never selects a checkpoint — it has seen no
  label; its job is to hand the labelled stages a better starting point.
- **Result** (test identities and queries identical to R3)

  | model | per-match Rank-1 | per-match mAP | whole-split Rank-1 | whole-split mAP |
  |---|---:|---:|---:|---:|
  | soccer_full (R4) | **26.13%** | **26.87%** | 15.73% | 15.85% |
  | soccer_full (R3) | 22.40% | 23.72% | 12.53% | 12.92% |
  | frozen CLIP | 14.13% | 16.69% | 6.40% | 8.08% |
  | NBA projection | 13.87% | 18.97% | 4.80% | 6.81% |

  Per match: Shakhtar-Malmo 31.63%, Man City-West Ham 27.01%, Frankfurt-Dortmund 22.33%,
  Leverkusen-Atletico 18.92% — the weakest match in R3 (14.56%) gained most.
  Bootstrap whole-split mAP difference vs frozen CLIP +7.78 pts (95% CI 3.10-12.32), vs NBA
  +9.05 pts (5.00-13.34).

  Reference aggregation, chosen on validation (`aggregation.json`):

  | rule | val Rank-1 | val MRR | test Rank-1 | test MRR |
  |---|---:|---:|---:|---:|
  | mean prototype | **42.09%** | **0.591** | 28.00% | 0.461 |
  | best single reference | 39.29% | 0.568 | 26.13% | 0.455 |
  | 2 prototypes | 39.03% | 0.579 | 29.07% | 0.472 |
  | 3 prototypes | 38.27% | 0.568 | 26.93% | 0.459 |
- **Visualisation** `reports/soccernet_full_v2/test_visualisation/`
- **Findings**
  - **Per-match Rank-1 rose from 22.40% to 26.13% and mAP from 23.72% to 26.87%** on the same test
    identities and queries. Whole-split rose further, 12.53% to 15.73% Rank-1.
  - **The gain came from the self-supervised stage, not from longer training.** The frozen-backbone
    head reached 18.20% validation mAP against R3's 16.15%, and R3's *complete* result after
    fine-tuning was 18.70%. Fine-tuning then added only +0.79 points and stopped improving
    after its second epoch. Self-supervised loss fell 37.7 to 14.3 over 8 epochs.
  - The confusions changed character. R3 confused players across teams inside a match
    (Strootman/El Shaarawy/Totti, all Roma). R4's top confusions are same-team pairs like
    Gonzalo Castro/Marcel Schmelzer and Jenkinson/Cresswell — the model is no longer being
    rescued or misled by colour, it is failing on people who genuinely look alike at 60 pixels.
  - **Mean prototype wins on validation**, as expected, and two prototypes is the best on test
    (29.07% against 28.00%) but was not chosen, because choosing by test score is choosing on
    test. The three rules sit within about 3 points, so a single centroid is not obviously
    blurring appearance modes at this gallery size.
  - Identity-level aggregation beats image-level ranking: 28.00% against 26.13%.
  - 117 of 277 misses still hold the true player in the top five of about twenty.
- **Disposition** current
- **Artifacts** `runs/soccernet_full_v3/`, `reports/soccernet_full_v2/training_r4.log`

### R5 — planned — gallery cleaning, aggregation, number awareness, tracklets

Two separate things, both arising from the R4 error sheets.

**Gallery cleaning (done, applies from the next freeze).** Every screen so far asks whether a
photograph shows one dominant person on a page that names the player; none asks whether it shows
the *same* person as that player's other photographs. `reference_outliers.py` scores each
reference by its mean cosine similarity to the rest of its player's gallery, using the trained
encoder, and renders anything below a threshold beside the references it most and least resembles.
At 0.60 it flagged 62 of 3085 references; **33 were confirmed wrong and excluded** — a goalkeeper
under an outfield player, a television presenter, a film poster, a mural, a cartoon, a grey
silhouette placeholder, a woman, coaching staff, spectators, and several players in the kit of a
club the named player never played for. The other 29 were legitimate outliers: a youth photograph,
a national-team kit, a former club. Nothing was excluded automatically.

These exclusions apply to the **next** freeze. R4's reported numbers stand as measured; re-scoring
R4 after cleaning a gallery whose defects were noticed in a test error sheet would be tuning on
test. What is judged is whether a photograph depicts the named player, never whether removing it
helps a score.

Two limits of the scan, both real:
- *It cannot see a defect that has company.* Anton Tinnerholm has 13 references of which 7 are one
  man in a checked shirt at a single Malmö farewell ceremony, so the ceremony outvotes the football
  photographs and the scan flags his *action* shot as the outlier. Whether that is him is genuinely
  unclear, but either way a gallery dominated by one off-pitch event is a poor reference set and
  its mean prototype cannot match a broadcast crop.

  A cap on how many references one source page may contribute was the first idea and was
  **rejected**: it throws away correct photographs to fix an aggregation artefact. The
  concentration only damages the mean prototype. It does not hurt max-over-references or
  image-level ranking at all, and in training every one of those photographs is still a valid
  positive. The fix that loses nothing is to **average within a source first, then across
  sources**, so one gallery page counts once instead of fifty-seven times — Vicente Guaita has 57
  references from a single forum page, Francesco Totti 21 from one club gallery, Tinnerholm 7 from
  one ceremony. Capping at three per page would have cut the gallery from 3,085 to 2,629; the
  source-weighted prototype costs nothing.
- *It needs a trained model*, so it can only run after a run, not before one.

**Tracklet quality head.** A small attention head rather than a temporal transformer:

    z_t = f(x_t)                       existing encoder, per crop
    q_t = W2 GELU(W1 z_t)              768 -> 128 -> 1, about 100k parameters
    w   = softmax(q / tau), tau ~ 2
    z_track = normalize( sum_t w_t normalize(z_t) )

trained end to end on the existing objective with `L = L_frame + lambda * L_track`, so the
per-frame embeddings cannot be allowed to rot because aggregation compensates. Guards against the
head collapsing onto one frame: 8 random frames per tracklet, temperature about 2, and a weak
entropy penalty. No quality labels — if emphasising a blurred crop hurts the identity objective,
its score falls.

**The data it needs does not exist in R4.** SoccerNet-v3/ReID gives an action frame plus linked
replay frames: multi-view snapshots of one instant, generally one crop of a player per view, not a
consecutive sequence. R4's self-supervised stage uses exactly that cross-view structure, which is
why it works there and why it cannot supply a tracklet. Temporal tracklets live in
**SoccerNet-Tracking** — 200 thirty-second clips plus one fully annotated 45-minute half across 12
games, with frame and track IDs — and **SoccerNet-Jersey** packages 2,853 tracklets built from it.
Neither is downloaded here; only the ReID archives are.

Coverage is the catch: Tracking covers 12 games against the 372 this dataset draws on, so annotated
tracklets cannot be had for most labelled players. The alternative is to build short tracklets
ourselves — anchor on a trusted SoccerNet-v3 box, track forward and back 1-2 seconds *within the
same camera shot*, take 8-16 sampled crops, and stop at a cut or a low-confidence association
rather than preserving the track aggressively. Each replay camera then yields its own
quality-weighted embedding and they are **not** merged before the loss. That keeps the tracking
risk bounded, since a short within-shot track anchored on an annotated box is far safer than a long
one, and it is also the form deployment needs.

Sequence: clean the gallery and switch to the source-weighted prototype at the next freeze; retrain; then build
tracklets, since the quality head is only worth training once there are tracklets to weight.

#### R5, continued — scope added after the R4 error sheets

**Committed: full SN-GSR-2025** (29.8 GB, ungated, downloading). Not SN-Jersey: its tracklets are
cropped for number reading and are therefore mostly back views, which is the opposite of what a
view-quality head needs to learn to prefer.

**Jersey-number awareness inside the one model.** The visual tower is all we use; CLIP's text
encoder is discarded. Bring it back frozen and align the existing 768-d embedding to the text
embedding of a caption like `a photo of soccer player number 7`, supervised by the shirt number we
already hold. No second model, no separate OCR stage, and the roster prior then applies for free:
once a number is read confidently and the team is known, the candidate set collapses from twenty
to one.

This is not in tension with erasing numbers in augmentation. The goal was never to ignore the
number but to stop depending on it — a model that reads it when legible and falls back on
appearance when not is exactly right, and the flat-fill augmentation conveniently provides a
trustworthy *negative*: a crop whose clothing was repainted flat definitely shows no number.

One honest caveat about the labels. Every query carries a number (11,143 of 11,143, 52 distinct)
because the number *is* the roster join key — it is a property of the player, not evidence that
the digits are visible in that crop. Supervising `this crop shows a 7` is therefore wrong whenever
the player is facing the camera or occluded. Either supervise only on flat-filled crops, where the
answer is known to be "none", plus crops a legibility gate accepts; or accept the noise and measure
it. Do not assume the label means visibility.

**Raising mAP by raising precision.** In rough order of cost:
- *k-reciprocal re-ranking.* Standard in ReID, needs no training, typically several mAP points.
  It exploits the fact that a true match's own neighbours contain the query.
- *Tracklet aggregation* — the quality head. Averaging even a handful of views of one player is
  worth more than any single-frame improvement, and 117 of R4's 277 misses already hold the true
  player in the top five.
- *Roster assignment.* Within a match each roster player is one person; solving the assignment
  jointly over a frame or a tracklet, rather than taking independent arg-maxes, removes the
  contradictions the same-team sheets are full of.
- *Confidence gating.* Commit a name only when the margin over the runner-up is large. This trades
  recall for precision explicitly rather than implicitly, and it is what deployment wants.

**Argentina-France demo.** The data is better than expected:
`/mnt/data/sports_reID/groundTruth_AllTracking_ARG_FRA_183303` holds 31 consecutive frames with 575
annotated boxes whose COCO categories are `<number> <player name>` — for example `10 Lionel Messi`,
`11 Angel Di Maria`. There is no explicit track id, but the category *is* the identity, so linking a
category across the 31 frames yields ground-truth tracklets with both name and number, from a real
broadcast outside SoccerNet entirely. `data/arg_fra_roster_probe_v1/queries` holds 511 crops already
cut from it, and `data/arg_fra_expanded_gallery_v1` holds the reference gallery for both squads.

The demo to build: stream the frames, accumulate each track's crops, run the quality-weighted
embedding, rank against that match's 22-player roster, and **name a player only when the margin
over the runner-up clears a threshold** — otherwise leave the box unlabelled. Render it as video
with the committed name, the ground-truth name and the margin, so a wrong commitment is visible.

The caveat that must ride with it: Argentina-France identities were deliberately forced into the
**training** split, because the earlier pilot exposed them. Messi, Mbappe, Di Maria, Rabiot and
Otamendi are all in training. This is a qualitative demonstration of the deployed system, not a
held-out measurement, and no number from it belongs in a results table.

**Repository organised.** `INDEX.md` now describes the layout. Superseded experiments, runs and
reports moved to `archive/superseded/` (8 GB); one-off install logs and browser probes to
`reports/legacy/`. Every current module was re-imported afterwards to confirm nothing broke.
Nothing was deleted: `archive/DELETION_CANDIDATES.md` lists about 130 GB of MuPNIT archives and
detector weights that nothing in the current pipeline reads, with what each one costs to lose.

## 15. In flight

State as of the last edit to this document, so the next session does not have to reconstruct it.

| item | state |
|---|---|
| R4 | complete and reported |
| R5 | complete; best validation of any run, second-best test, difference from R4 not distinguishable |
| Best deployable model | `runs/soccernet_full_v3/best.pt` (R4), under the old evaluation |
| Evaluation | rebuilt for R6: 12 matches per split, one-reference eligibility; old test numbers do not carry over |
| Quality head | trained and evaluated; **not adopted** — aggregation helps, the learned weighting is not distinguishable from a plain mean |
| Gallery cleaning | 33 references excluded; applied at the R5 freeze |
| Source-weighted prototype | implemented in `aggregate.py` as rule `source-mean`; compared on validation with the others |
| SN-GSR-2025 | unpacked, 59 GB — 164 sequences of 750 frames at 25 fps, 3,484 player tracklets, 1.76 M boxes, from **9 source games** |
| Paced Wikimedia collector | running, deficit-ordered, about 48 images an hour |
| Repository | organised; `INDEX.md` is the map, `archive/DELETION_CANDIDATES.md` lists ~130 GB nothing reads |

Decisions taken in discussion that are easy to lose:

- **No reference cap.** See R5 above. Fix the aggregation, not the data.
- **No test-time augmentation.** It would only be worth adding if it measurably improved test
  performance, which has not been tested. Query augmentation stays a training-time device.
- **Full match video is NDA-gated.** `SoccerNet_raw_HQ` and `SoccerNet-Tracking-RAW-Video` are
  `gated=manual` on HuggingFace; access is a form the dataset owners approve. Ungated third-party
  re-uploads of the 500 videos exist and are deliberately **not** used, because they exist to route
  around that gate. `SN-GSR-2025`, `SN-Tracking-2023` and `SN-Jersey-2023` are genuinely ungated
  and were downloaded on that basis.
- **Tracklet coverage is limited and that is acceptable**, because the quality head is what the
  tracklet data is for. It learns which views are trustworthy, which is identity-agnostic and needs
  many frames rather than many people. Identity supervision stays on the 410 players and 372 games
  we already have. Our own multi-view data cannot substitute: of 72,127 persons, 53,411 have
  exactly two views and only 5,755 have four or more, which is far too thin to train an attention
  head over eight frames.
- **Argentina-France is a demonstration, not a measurement.** Those identities were forced into
  training on purpose; Messi and Otamendi are in the train split.

### R5 — completed — gallery cleaned, source-weighted prototype

- **Splits** train 283/10,366/2,149, val 64/392/450, test 61/375/453 — same test identities and
  queries as R3 and R4
- **Changed from R4** 33 confirmed-wrong references removed; `source-mean` added to the
  aggregation comparison. Training recipe identical, so this reads as gallery quality alone.
- **Training** selected finetune epoch 3, validation mAP 19.60% — the best validation of any run
- **Result**

  | model | per-match Rank-1 | per-match mAP | whole-split Rank-1 | whole-split mAP |
  |---|---:|---:|---:|---:|
  | R5 | 23.73% | 25.44% | 14.13% | 14.26% |
  | R4 | 26.13% | 26.87% | 15.73% | 15.85% |
  | frozen CLIP | 14.13% | 16.80% | 6.93% | 8.30% |
  | NBA projection | 14.40% | 19.06% | 5.60% | 7.03% |

- **Findings — the measurement is the result**
  - **R5 has the best validation of any run and the second-best test.** Validation mAP rose
    18.99% to 19.60% while per-match test Rank-1 fell 26.13% to 23.73%. Paired on the identical
    375 queries the difference is **-2.40 points with a player-clustered 95% interval of
    [-12.6, +6.6]**, and R5 wins in 33% of resamples. Removing 33 references out of about 3,850
    is not measurable with four test matches. An earlier note in this log claiming the cleaning
    was worth +0.61 was read off validation before test existed and is withdrawn.
  - **Validation ranks the aggregation rules wrongly too.** `source-mean` is the best rule on
    test at 28.80% Rank-1 against `mean` at 25.33%, and validation placed it fourth of five.

    | rule | val Rank-1 | val MRR | test Rank-1 | test MRR |
    |---|---:|---:|---:|---:|
    | mean | 40.05% (chosen) | 0.585 | 25.33% | 0.442 |
    | source-mean | 38.27% | 0.574 | **28.80%** | 0.453 |
    | max | 36.73% | 0.555 | 23.73% | 0.433 |
    | proto-2 | 40.05% | 0.578 | 26.67% | 0.451 |
    | proto-3 | 40.82% | 0.576 | 26.93% | 0.449 |

    The rule is still chosen on validation, because choosing it on test is choosing on test. But
    the disagreement is the point.
  - **Two independent signals now say the evaluation cannot rank changes this size.** 64
    validation players over 6 matches and 61 test players over 4 matches, with per-match Rank-1
    varying threefold between matches. Single-run differences under about 5 points carry no
    information, and treating them as evidence has already produced one withdrawn claim in this
    log. Before more modelling: more evaluation matches, and repeated seeds to measure run-to-run
    variance directly.
  - R5 still beats both baselines with intervals excluding zero: +5.96 mAP points over frozen
    CLIP (95% CI 2.08-9.72) and +7.23 over the NBA projection.
- **Disposition** current dataset; **R4 remains the better test result** and its checkpoint
  `runs/soccernet_full_v3/best.pt` is the one to deploy until something beats it measurably
- **Artifacts** `runs/soccernet_full_v4/`, `reports/soccernet_full_v2/training_r5.log`

#### What SN-GSR-2025 actually contains

Unpacked: **164 sequences**, each 750 frames at 25 fps — thirty seconds — with per-box
`track_id`, and `attributes` giving role, jersey number and team side. **3,484 player and
goalkeeper tracklets**, 1.76 M boxes, median **532 frames per tracklet**, 3,430 tracklets with at
least 64 frames, and 82.1% of boxes carrying a readable jersey number.

This confirms both halves of the earlier judgement. It is superb for the quality head — thousands
of tracklets, hundreds of frames each, which is precisely what our own multi-view data cannot
supply, where 53,411 of 72,127 persons have exactly two views. And it is useless as identity
supervision: **9 source games**, against the 372 our labelled set draws on, and it gives
jersey number and team rather than names. Train the head here on tracklet-internal supervision,
which needs no names, and leave identity supervision where it is.

GSR's jersey attribute is also the honest per-crop visibility signal the labelled set lacks — here
17.9% of boxes are explicitly marked unreadable, whereas in our manifest every query carries a
number because the number is the join key. **It will not be used to train a legibility gate.** One
model, not a stack of task-specific heads: a separate supervised classifier for "is the number
readable" would be a second model in all but name, and its output would have to be fused by hand.

What stays, and why it is consistent with that:

- *The quality head* has no supervision of its own. It is a 768-128-1 MLP trained only by the
  identity objective, so it discovers that blurred, tiny, occluded and back-turned views deserve
  less weight because emphasising them hurts retrieval. Nothing is labelled "good frame".
- *Number awareness* goes through CLIP's frozen text encoder, aligning the existing embedding to
  captions like `a photo of soccer player number 7`. That keeps numbers inside the one shared
  space rather than in a parallel classifier whose logits someone then has to combine.

GSR's jersey attribute is still useful — as a held-out check on whether the model learned
legibility implicitly, by asking whether the quality head's weights correlate with it. Measuring a
correlation is not the same as training on it.

### R6 — running — the evaluation rebuilt

R5 produced two independent demonstrations that the evaluation could not rank the changes being
made to it, so this run changes the measurement rather than the model. **Training recipe is
identical to R4 and R5.**

- **Evaluation eligibility relaxed from three references to one.** Requiring three both shrank the
  test set to four matches and flattered it: a roster in deployment supplies whatever photographs
  exist for each of the twenty-two, often one. Training still requires three, where several
  references genuinely help. This alone took the eligible pool from 412 players to 983, and the
  matches holding ten or more eligible players from 66 to 222.
- **Twelve matches per evaluation split instead of four and six.**

  | split | players | queries | references | matches | before |
  |---|---:|---:|---:|---:|---|
  | train | 265 | 9140 | 2123 | 286 | 283 / 10,366 |
  | val | 115 | 1043 | 324 | 12 | 64 / 392 / 6 matches |
  | test | 182 | 1448 | 818 | 12 | 61 / 375 / 4 matches |

  Test is three times the players, nearly four times the queries and three times the matches, for
  a 6% cut in training identities. Match-level clustering was the dominant variance term — per-match
  Rank-1 varied threefold across four matches — so tripling the matches matters more than the query
  count does.
- **`compare_runs.py`** now answers "is this difference larger than the noise" directly: paired over
  the queries two runs share, resampled clustered by player, reporting whether the interval excludes
  zero. Pairing matters because both runs answer the same queries; comparing totals independently
  discards that and widens the interval for nothing.
- **The leak assertion earned its keep.** Changing the evaluation matches put 3,841 self-supervised
  persons inside held-out matches; `train_v2` refused to start and named the number. The pairs were
  rebuilt against the new split — 24 matches now excluded, 68,985 persons, 164,491 crops.
- **Old test numbers do not carry over.** R6's test players overlap the training sets of R4 and R5,
  so those checkpoints cannot be re-scored here without leaking. R1 to R5 stand as measured under
  the old protocol and are not comparable to anything from R6 onward. Expect lower absolute
  numbers: a one-photograph gallery is a harder and more honest gallery than a three-photograph one.

#### R6 — completed — the rebuilt evaluation works

| model | per-match Rank-1 | per-match mAP | whole-split Rank-1 | whole-split mAP |
|---|---:|---:|---:|---:|
| R6 | **25.55%** | **28.54%** | 12.09% | 10.78% |
| frozen CLIP | 14.85% | 19.12% | 3.87% | 4.44% |
| NBA projection | 14.99% | 23.18% | 2.35% | 4.44% |

Selected finetune epoch 1, validation mAP 17.55%.

**The interval is the point.** `compare_runs.py` puts R6 at 25.55%
[20.62, 30.91] — about five points wide, where R4 against R5 spanned
[-12.6, +6.6]. Differences of roughly five points are now detectable; nothing under thirteen was
before. Per-match scores still range 11.9% to 33.6%, so one match still means nothing; averaging
twelve instead of four is what bought the narrower interval, confirming match clustering was the
dominant variance term.

Two things the bigger evaluation immediately corrected:

- **The absolute number barely moved** — 25.55% against R4's 26.13% — despite roughly half the
  test players now having one or two reference photographs instead of three or more. The old
  three-reference bar was flattering the result less than expected.
- **The aggregation rules collapsed together.** They now span 25.00% to 26.52% and validation
  ranks them nearly correctly. Under R5's small evaluation `source-mean` beat `mean` by 3.5 points
  on test while validation ranked it fourth; that gap was noise, not an effect of source weighting.
  Recorded here because it was over-read at the time.

#### Quality head — built, evaluated, **not adopted**

*First attempt, on GSR track against track: abandoned after two epochs.* Separating eight tracks
inside one thirty-second sequence reaches 99% with a plain mean, so weighting had no headroom to
help and the softmax had every freedom to concentrate. It concentrated to 4x and scored five
points **below** the mean it was meant to improve. A saturated task cannot teach quality.

*Second attempt, on the deployment objective:* several broadcast views of a player against a clean
reference photograph, negatives being other players of the same match, groups formed per player
per match. Here the plain mean sits at 39%, so there is room to work in. Trained 12 epochs on the
labelled train split with the backbone frozen, `L_frame + L_track` and an entropy penalty.

Test results, per match against that match's roster, paired and clustered by player:

| grouping | single crop | plain mean | size weighted | quality weighted |
|---|---:|---:|---:|---:|
| per action (1.90 views) | 26.64% | 29.40% | 20.21% | 30.58% |
| per match (7.91 views) | 28.96% | 34.43% | 35.52% | 36.61% |

- **Aggregation is a real gain.** Quality weighted beats a single crop by
  +3.94 [+1.74, +6.16] per action and
  +7.68 [+1.09, +14.29] per match. Both exclude zero.
- **The weighting is not distinguishable from a plain mean.** +1.19
  [-0.40, +2.86] per action, +2.20 [-2.19, +6.56] per match.
- **What it actually learned is close to crop resolution.** Relative weight correlates
  +0.52 with log crop pixels and only -0.11 with aspect ratio, so it reads
  scale and not posture or occlusion. `reports/soccernet_full_v2/quality_weights/` shows this
  directly: Hakan Balta's 291x575 view takes 0.854 of the weight while his 78x111 and 66x120 views
  take almost none.
- **But it has not reduced to crop size.** A hand-written softmax over standardised log area
  matches it per match (35.52% against 36.61%, indistinguishable) and **collapses per action**
  to 20.21%, nine points below the plain mean, because with two views the standardisation
  forces a near one-hot split and betting on the larger crop is worse than averaging. The head,
  restrained by the entropy penalty, avoids that. The correlation is a tendency, not the rule.

**Conclusion.** Aggregation stays; the learned head does not. `infer.py` should default to a plain
mean over the tracklet. Settling the head properly needs an aggregation task where bad views
genuinely dominate, or several seeds to separate +1.5 points from zero.

---

## Whole-body references: measured, and mostly a dead end

**2026-09-17.** The portrait/body worry was, until now, inferred from which websites a photograph
came from. Every stored reference is a dominant-person crop, so the crop's own height/width *is*
the shape of the detected player, and that can be measured directly. About 1.0-1.4 is a
head-and-shoulders press portrait; above 2.0 is a standing whole player.

| set | median person aspect | share above 2.0 |
|---|---:|---:|
| Argentina-France track queries | 2.27 | — |
| SoccerNet test queries | 2.00 | — |
| SoccerNet test references | 1.34 | 14% |
| Argentina-France v1 gallery, cropped | 1.35 | 26% |

The gap is real and it is not confined to the demonstration gallery: our own frozen test split has
it too. Three things were then tried against it.

**Does selecting taller references help? Not distinguishably.**
`aspect_study.py` rebuilds each identity's prototype from the taller half of that identity's own
references and from the shorter half — equal counts in both arms, so only shape differs.

| arm | references | Rank-1 | MRR |
|---|---:|---:|---:|
| all | 818 | 25.00% | 42.39% |
| taller half | 405 | 25.14% | 42.04% |
| shorter half | 405 | 22.51% | 39.33% |

Taller minus shorter is **+2.62 Rank-1 [-1.93, +7.62]** and **+2.70 MRR [-1.76, +7.51]**,
player-clustered bootstrap: the right sign, not separable from zero. Taller-half alone matches all
818 references using half of them, which is the only clean practical consequence — the squattest
half of the gallery is carrying nothing.

**Can the demonstration gallery be rebuilt quickly? Yes, but only for the famous.**
Three collection routes, measured:

| route | rate | outcome |
|---|---|---|
| Wikimedia direct | ~6 images, then a 600s `Retry-After` | 287 candidates ≈ 8 hours |
| `wsrv.nl` image proxy | 36 images in 25s, then upstream 429 through the proxy, negative-cached | dead |
| free news photo galleries | 12 pages in seconds, unthrottled | 59 images, 18 of Messi, 16 of Mbappe, 6 players with none |

`demo_web.py` mines match galleries: one page names many players in its captions, and a photograph
counts for a player only when its caption names that player and no other on the roster. The 2022
final is excluded by hand — it is the clip the demonstration runs on. Screening had to be relaxed
for this source (`relaxed()`, second-person ratio 0.60 instead of 0.35): the portrait-era rule
rejects any photograph with a comparable second person, which is most action photography, and it
threw away 50 images before the change.

**Did it help? No.** `demo_gallery.py build` merged 26 Commons and 17 news-gallery references with
121 v1 portraits as fallback, keeping all 21 identities.

| gallery | Rank-1 | truth in top 3 | median rank |
|---|---:|---:|---:|
| v1 portraits, 168 images | 4/21 | 10/21 | 4 |
| rebuilt, 164 images (43 new) | 5/21 | 10/21 | 4 |

One track's worth of movement at n=21 is noise.

**Why the rebuild barely moved the shape.** The new references have median person aspect **1.46**
and 23% above 2.0, against 1.35 and 26% for the portraits they replaced. A news action photograph
is not a standing whole player: it is a cropped, bent, running player. The source label
("action photo" versus "portrait") does not predict the thing we actually wanted.

**Disposition.** The aspect gap is documented and real; no intervention against it has been shown
to pay. Do not spend collection budget on chasing it. The one change worth keeping is negative:
the squattest half of the reference gallery can be dropped at no measured cost.

**Artifacts.** `experiments/soccernet_full_v2/aspect_study.py`, `demo_gallery.py`, `demo_web.py`;
`reports/soccernet_full_v2/aspect_study.json`;
`data/arg_fra_expanded_gallery_v1/{gallery_body.jsonl,gallery_v1_cropped.jsonl,web_candidates.jsonl}`;
`reports/soccernet_full_v2/demo_arg_fra_body/`. The frozen v1 gallery is untouched.

---

## Where the loss actually is

**2026-09-17.** Asked what the right fix is, rather than defending the last one.
`error_anatomy.py` splits per-match retrieval on the frozen test split into the two questions it
conflates — which team, and which of that team's eleven — and prices the levers against each other.

Per-match retrieval, 1,448 test queries, mean roster 15.9 with 8.4 teammates:

| quantity | value | chance |
|---|---:|---:|
| Rank-1 | 25.0% | 6.3% |
| team identified | 73.1% | 53.0% |
| Rank-1 if the team were known perfectly | 34.1% | 11.9% |
| when wrong, the top pick is a teammate | 64.2% | 50.4% |
| **Rank-1 if the shirt number were read** | **80.8%** | — |

Grouped into the 183 player-by-match tracklets the model is actually deployed on (median 6 crops),
a plain mean gives **Rank-1 32.2%, top-3 57.4%, median rank 3**, against a number-reading ceiling
of **78.1%**.

Query size is not the binding constraint. Across a 32-fold range of crop area, Rank-1 moves from
22.9% (median 5,049 px) to 30.9% (median 162,700 px) — eight points for thirty-two times the
pixels. Broadcast against replay is 23.3% against 26.4%.

**The comparison that matters.** Every appearance-side lever measured so far moves a few points and
mostly cannot be separated from zero: reference shape +2.6 [-1.9, +7.6], hubness centring +1.4
[-1.8, +4.7], learned view weighting +1.2 [-0.4, +2.9], aggregation rule a couple of points. The
shirt number is worth **+46 points of ceiling at tracklet level**. Nothing else on the table is
within an order of magnitude, and the reason is structural: within a match both candidates wear the
same kit, at the same distance, and the number is the only field on a player that is unique by
construction.

**What the ceiling does and does not assume.** It assumes a perfect reader and takes the number
from roster metadata, which we already hold — 78.1% is simply the share of tracklets whose number
is unique on their roster, so no photograph of the number is needed on the reference side at all.
It does *not* assume the number is legible: SoccerNet-GSR annotates jersey number per track and not
per detection (82.0% of its 3,260 player tracks carry one, identically at every "at least k views"
threshold, which is what gives the constancy away), so per-frame legibility is not measurable from
what we have. The realised gain is bounded by legibility and will be well under the ceiling. A
tracklet needs only one legible view, though, and the median tracklet has six.

**Disposition.** Stop spending on reference photographs. The next experiment is the number channel
inside the one model, as R5 planned: CLIP's frozen text encoder, captions of the form
`a photo of a soccer player wearing number 7`, the query embedding scored against both the
reference-photo prototype and the roster-metadata text prototype, and evaluation at tracklet level
where a single legible view decides the track. Appearance becomes the fallback for tracks whose
number is never visible, which is the right shape for the problem rather than the only shape we had.

**Artifacts.** `experiments/soccernet_full_v2/error_anatomy.py`,
`reports/soccernet_full_v2/error_anatomy.json`.

---

## The shirt number, inside the one model

**2026-09-17.** Built as one embedding space, not two models. One frozen CLIP text anchor per
number (`numbers.py`, prompt ensemble over 0-99, cached to `weights/number_text.pt`), placed into
the projection space by a single 590k-parameter linear map that lives in `ReIDModel` beside the
projection head. Each identity's prototype becomes one vector:

```
p = normalize(alpha * photo_prototype + (1 - alpha) * number_anchor)
```

Retrieval stays a single dot product against a single prototype per player. There is no second
score to combine, no gate, no separate recogniser — the embedding that has to match the photograph
is the same embedding that has to match the number. `alpha` is chosen on validation MRR and applied
unchanged to test.

### Does CLIP already read shirt numbers? Yes.

Frozen CLIP ViT-L/14 image features against the raw text anchors, zero-shot, on the test queries:

| | top-1 | control |
|---|---:|---|
| over all 100 numbers | **25.6%** | majority class 8.5%, shuffled labels 1.7% |
| restricted to the ~14 numbers on that match roster | **42.5%** | chance 7.2% |

Not one lucky class: recall is 23.6% on number 5 (n=123), 42.2% on 2, 48.5% on 10.

### Our projection head was throwing that signal away

Same adapter, same training, fitted on features taken before and after the projection head:

| features the adapter reads | within-roster number top-1 (test) |
|---|---:|
| `encode_base`, pre-projection | **36.6%** |
| after the projection head | **21.6%** |

This is not an accident, it is something we did on purpose: `QueryAugment` painted the shirt flat
— erasing the digits — on **half** of all training crops, to stop the identity loss reading the
number as a shortcut. It worked, and it suppressed the cue with the largest headroom in the
problem. The joint trainer drops that probability to 0.2 and masks the number loss on any crop
whose digits it did just erase, so the model is never asked to invent a number it cannot see.

### What the channel is worth, with the head still suppressing it

Adapter fitted on train queries against the frozen `soccernet_full_v5` model, `alpha` chosen on
validation, evaluated on test:

| | photo only | fused | delta | 95% CI (player-clustered) |
|---|---:|---:|---:|---|
| per crop, Rank-1 | 25.00% | **30.25%** | +5.25 | [+0.94, +9.59] |
| per crop, MRR | 42.39% | **47.89%** | +5.51 | [+2.09, +8.97] |
| per tracklet, Rank-1 | 32.24% | **36.07%** | +3.83 | [-2.72, +10.44] |
| per tracklet, MRR | 49.52% | **54.82%** | +5.31 | [+0.98, +9.90] |
| per tracklet, top-3 | 57.38% | **66.67%** | +9.29 | — |

Median rank falls from 4 to 3 per crop and from 3 to 2 per tracklet. Number-only prototypes
(`alpha`=0) score 19.5% per crop, *below* photo-only — the fusion beats both ends, which is what a
genuinely complementary channel looks like rather than one signal dominating.

**Three of these four intervals exclude zero.** That is the first lever in this project that does.
For comparison, on the same test split: reference shape +2.6 [-1.9, +7.6], hubness centring +1.4
[-1.8, +4.7], learned view weighting +1.2 [-0.4, +2.9].

### Caveats

- The adapter here was fitted against a **frozen** re-identification model, so this measures the
  channel with the image head still actively discarding number information. The joint run should
  be better, not worse, but that is a prediction and not a result.
- `alpha` was chosen on validation, and validation and test disagree slightly on where the optimum
  sits (0.2 per crop and 0.3 per tracklet on validation; test prefers 0.3-0.5 per tracklet). The
  reported numbers use the validation choice.
- The number comes from roster metadata, which we already hold, so nothing has to be visible in a
  reference photograph. What must be visible is the number on the **query** crop, and per-frame
  legibility is still unmeasured — SoccerNet-GSR annotates jersey per track, not per detection.

### What changed in the code

`src/soccer_gif_reid/model.py` (`NumberAdapter`, `ReIDModel(numbers=...)`, `build_model(numbers=...)`,
`from_checkpoint` detects it), `experiments/soccernet_full_v2/numbers.py` (anchors),
`number_channel.py` (fit and sweep against a frozen model), `augment_v2.py` (`QueryAugment.erased`),
`train_v2.py` (`--number-weight`, `--flat-kit`, number term in `run_epoch`, adapter in the optimizer).
`--number-weight 0` reproduces the previous behaviour exactly.

**Artifacts.** `reports/soccernet_full_v2/number_channel.json`, `weights/number_text.pt`,
`runs/soccernet_full_v5/number_cache/`.

---

### R7 — completed — number channel trained inside the model — 2026-09-18

- **Manifest** `4f9de598127cd103` (unchanged from R6; only the model and training changed)
- **Splits** train 265/9140/2123, val 115/1043/324, test 182/1448/818
- **Evaluation** per-match, and per player-by-match tracklet (183 groups, median 6 crops)
- **Changed from previous** shirt number as text anchors inside the one embedding space:
  `NumberAdapter` (590k parameters) in `ReIDModel`, number cross-entropy at weight 2.0 on the same
  query embedding the identity loss uses, number-erasing augmentation cut from 0.5 to 0.2, and the
  number term masked on any crop whose digits the augmentation erased.
- **Training** ssl 8, head 30, finetune 15; selected finetune epoch 1, validation mAP 16.86%
  (v5: 17.55%). Resumed once at head epoch 3 with 2 dataloader workers instead of 6 after the
  machine ran low on memory; nothing else changed.
- **Result** test, alpha chosen on validation MRR and applied unchanged:

| | v5 photo only | v6 photo only | **v6 fused** |
|---|---:|---:|---:|
| per crop Rank-1 | 25.00% | 26.38% | **38.67%** |
| per crop MRR | 42.39% | 43.73% | **54.93%** |
| per tracklet Rank-1 | 32.24% | 32.79% | **52.46%** |
| per tracklet top-3 | 57.38% | 53.55% | **79.78%** |
| per tracklet MRR | 49.52% | 49.67% | **67.96%** |
| per tracklet median rank | 3 | 3 | **1** |

  Player-clustered bootstrap against the previous best model (v5, photographs only):
  per crop Rank-1 **+13.67 [+8.73, +18.78]**, MRR **+12.54 [+8.75, +16.53]**;
  per tracklet Rank-1 **+20.22 [+10.99, +29.12]**, MRR **+18.44 [+11.78, +25.00]**.
  Against v6's own photo-only prototypes: +12.29 [+7.77, +16.93] and +19.67 [+10.99, +28.42].
  Every interval excludes zero.

- **Number reading** inside the model, test: 32.9% within the match roster (chance 7.2%) and 23.6%
  over all 100 numbers (chance 1%). The post-hoc adapter on the frozen v5 head reached 21.6% and
  13.7%. Training the channel jointly, and erasing digits less often, is what closed that gap.
- **Findings**
  - **Number-only prototypes now beat photographs.** 37.16% against 32.79% Rank-1 per tracklet.
    Before joint training they were far worse (19.5% against 25.0% per crop). The fusion still
    beats both ends, so the channels stay complementary rather than one replacing the other.
  - **Photo-only retrieval paid about a point.** v6 tracks v5 roughly 0.5-1.3 points lower on
    validation mAP throughout the head stage, which is the capacity the auxiliary loss takes.
    Cheap at the price.
  - **Selection is still on photo-only validation mAP**, a metric the number channel cannot help
    and can only hurt. That biases *against* the change, so the gains above are understated, not
    inflated. Selecting on fused validation MRR is the obvious next refinement.
  - **Where alpha sits.** Test prefers 0.3-0.4 per crop and 0.3-0.4 per tracklet; validation chose
    0.4 and 0.2. The per-tracklet number is therefore a little below the best available (52.46%
    against 55.19% at alpha 0.4), which is the honest cost of choosing on validation.
- **Visualisation** `reports/soccernet_full_v2/test_visualisation/` (photographs only, the
  shared folder the visualiser writes) and `reports/soccernet_full_v2/test_visualisation_fused/`
  (what the deployed fused prototypes actually predict)
- **Disposition** kept as current. This is the largest measured gain in the project, and the first
  from a direction outside appearance.
- **Artifacts** `runs/soccernet_full_v6/`, `reports/soccernet_full_v2/number_channel.json`,
  `weights/number_text.pt`

---

## The number channel on the Argentina-France clip, and why it failed there

**2026-09-25.** The v6 demo (`--alpha 0.2`, the per-tracklet value validation chose) named 17 of
21 tracks at a validation-calibrated threshold of 0.035 (91.4% precision on validation) and was
right on **4 — 23.5% precision**. Nahuel Molina absorbed 8 tracks and Ousmane Dembélé 6.
`number_transfer.py` separates the causes.

**The clip is outside the range the channel was validated on.** Number reading on SoccerNet test,
within the match roster, by crop area:

| crop pixels | test crops | number read correctly |
|---|---:|---:|
| < 2,000 | 6 | 0% |
| 2,000-4,000 | 75 | 16.0% |
| 4,000-8,000 | 418 | 27.8% |
| 8,000-16,000 | 464 | 38.8% |
| 16,000-64,000 | 174 | 42.5% |
| > 64,000 | 311 | 30.2% |

Clip crops are median **1,500 px**; test crops are median 10,399. Six test crops are as small as a
typical clip crop. On the clip the model reads **12.3%** within the roster's 17 distinct numbers
(chance 5.9%), and 4 of 21 tracks by their mean embedding.

**Where it cannot read, it guesses the prior.** Numbers 5, 11 and 10 take 297 of the clip's 511
reads. That is the training signal v6 gave: crops whose digits the augmentation erased were
*masked out* of the number loss, so the model was never told that "no readable number" is an
answer. A confident guess at a common number is what it learned to do.

**The commit threshold cannot survive that.** Calibrated on legible validation crops, a margin of
0.035 means something; on the clip the same margin is reached by confident wrong reads.

**Full-clip ranks** (mean of all views, no streaming; 21 tracks, so each track is 4.8 points):

| prototype | raw Rank-1 | centred Rank-1 | centred top-3 |
|---|---:|---:|---:|
| photographs only (alpha 1.0) | 5 | 6 | 12 |
| alpha 0.8 | 4 | **10** | 13 |
| alpha 0.6 | 6 | 8 | 14 |
| alpha 0.4 | 6 | 5 | 15 |
| alpha 0.2 (validation's choice) | 5 | 4 | 12 |

A light number weight helps on these crops and a heavy one hurts, which is the shape the
legibility table predicts. Alpha is **not** being chosen on the clip: that would tune on the thing
being demonstrated.

**The roster itself is harder than a club roster.** Only 13 of the 21 have a number no one else on
the roster wears: De Paul and Griezmann are both 7, Álvarez and Giroud 9, Messi and Mbappé 10,
Di María and Dembélé 11. The number can narrow those four pairs to two, not to one.

### R8 — running — `soccernet_full_v7`

Two changes, each behind a flag so v6 stays reproducible:

- **`--number-unknown uniform`.** An erased-number crop is trained toward equal probability over
  all 100 number anchors instead of being skipped. An embedding equidistant from every anchor makes
  the number half of every prototype score identical, so the number stops deciding anything exactly
  when it cannot be read. Same loss, same embedding, no gate, no second head.
- **`--select fused`.** Checkpoints are selected on per-match fused validation MRR, alpha swept on
  validation each epoch, instead of whole-split photo mAP, which the number channel cannot move.
  The photo-selected checkpoint is kept too (`best_photo.pt`), so the run separates the two
  effects: `best_photo.pt` against v6 isolates the uniform target under the same selection rule,
  and `best.pt` against `best_photo.pt` isolates the selection rule.

Everything else matches v6: number weight 2.0, flat-kit 0.2, 8/30/15 epochs, seed 42.

**What this cannot fix.** The uniform target is taught on *erased* crops — plain painted shirts.
Whether it transfers to shirts whose digits are simply too small to resolve is the open question.
The direct way to teach that is to show the model small crops, which is the resolution-degradation
augmentation turned off earlier at the user's request; it stays off.

Also this session: the curling arena servers on ports 8030 and 8031 were stopped at the user's
instruction to free GPU memory (7.7 GB) and RAM (~27 GB).

**Artifacts.** `experiments/soccernet_full_v2/number_transfer.py`,
`reports/soccernet_full_v2/number_transfer.json`, `reports/soccernet_full_v2/demo_arg_fra_number/`,
`runs/soccernet_full_v7/` (in progress).

### R8 — completed — `soccernet_full_v7` — 2026-09-25

- **Manifest** `4f9de598127cd103` (unchanged)
- **Changed from previous** `--number-unknown uniform` and `--select fused`; everything else as v6.
- **Training** ssl 8, head 15 (early-stopped), finetune 12. Fused selection chose finetune epoch 5
  (validation fused MRR 65.46% at alpha 0.4); the photo rule would have chosen head epoch 6
  (validation fused MRR 61.50%).
- **Result**, test, alpha chosen on validation:

| | v6 | v7 photo-selected | **v7 fused-selected** |
|---|---:|---:|---:|
| per crop Rank-1 | 38.67% | 35.98% | **41.09%** |
| per crop MRR | 54.93% | 52.40% | **56.59%** |
| per tracklet Rank-1 | 52.46% | 46.45% | **54.10%** |
| per tracklet top-3 | 79.78% | 74.32% | **81.97%** |
| per tracklet MRR | 67.96% | 62.65% | **69.82%** |
| number read within roster | 32.9% | 26.0% | 31.4% |

  v7 minus v6, player-clustered: per crop Rank-1 +2.42 [-0.64, +5.65], MRR +1.66 [-0.77, +4.16];
  per tracklet Rank-1 +1.64 [-4.95, +8.24], MRR +1.86 [-2.34, +6.11]. **Not distinguishable.**
  Error sheets: 327 queries fixed by the number, 106 broken; same-team confusions 624 → 441.

- **Findings**
  - **Selection on the deployment metric matters.** Inside one run, the fused-selected checkpoint
    beats the photo-selected one by +5.1 Rank-1 per crop and +7.7 per tracklet on test. Photo mAP
    stopped the head stage at epoch 6, before the number channel had matured (26.0% against 31.4%
    number reading). Fused selection is kept.
  - **The uniform target did not reach its own objective.** On erased crops the model's top number
    still carried 16.8% probability at the end of training, against 1% for undecided, and the term
    rose from 4.67 to 5.56 over training. The identity loss (roughly 16-70) and the readable-number
    term, which sharpens the shared temperature, outweigh it.
  - **And it did not transfer to the clip.** Clip number reading 12.3% → 13.9% within the roster;
    per-track reads 4 → 6 of 21; the read histogram is no flatter (5, 10, 24, 11 take 347 of 511).
    The demo at validation's alpha 0.4 named 18 of 21 tracks at **27.8% precision** (v6: 23.5%),
    with Messi, Mac Allister and Dembélé as the attractors.
  - Because the photo-selected v7 checkpoint comes from a much earlier point in training than v6's,
    `best_photo.pt` against v6 does **not** cleanly isolate the uniform target as planned. What
    can be said is that nothing measured shows it helping.
- **Where that leaves the demonstration.** The clip's crops (median 1,500 px) sit below the size at
  which the number is readable at all (0/6 test crops under 2,000 px, 10.7-16.0% at 2,000-4,000),
  and the model has no way to know it cannot read them. Painted-over shirts did not teach that. The
  direct way is to train on small crops, the resolution-degradation augmentation that is off by the
  user's choice. That is the user's decision, not a default to change quietly.
- **Visualisation** `reports/soccernet_full_v2/test_visualisation_fused_v7/`,
  `reports/soccernet_full_v2/demo_arg_fra_number_v7/`
- **Disposition** kept as current for fused selection; the uniform target is inert as implemented.
- **Artifacts** `runs/soccernet_full_v7/{best.pt,best_photo.pt}`,
  `reports/soccernet_full_v2/number_channel_soccernet_full_v7_{best,best_photo}.json`,
  `reports/soccernet_full_v2/number_transfer_{v6,v7}.json`

### R9 — running — `soccernet_full_v8` — 2026-10-20 — strict paired-data-only restart

- Found and documented a previously missed identity leak in auxiliary pretraining. Of v7's
  first 40,000 auxiliary groups, 1,508 resolve to held-out identities: 319 groups covering
  48 validation players and 1,189 groups covering 130 test players. Anonymous groups remain
  unresolved. Earlier match-disjoint checks did not establish player separation.
- Corrected future auxiliary filtering to exclude whole matches with held-out roster IDs or
  normalized names and unknown rosters; 5,817/68,985 groups over 28 matches remain. V8 disables
  the auxiliary stage altogether and starts from pretrained CLIP with fresh heads.
- Frozen paired manifest stays `4f9de598127cd103…`: 265/115/182 players in train/val/test,
  9,140/1,043/1,448 queries and 2,123/324/818 references. Stored/source image hashes checked.
- Training: 0 auxiliary epochs, head max 30, fine-tune max 15, 500 batches/epoch, batch 16,
  two workers, seed 42. Number weight 2.0, flat-kit 0.2, unknown uniform, fused validation MRR
  selection. No added quality degradation. Early-stopping patience now follows the selected
  metric independently of photo-only improvement.
- Independent v7 recount agrees on fused ranks, but its scores now carry the auxiliary leakage
  caveat. Whole-split/photo and fused/prototype protocols remain separately reported. Pooled
  groups are ground-truth identity-by-match, not real tracker output. The existing test was
  reused during development and is not a newly untouched final test for v8.
- Regression checks: 8 tests passed, including held-out identities in another auxiliary match,
  normalized-name aliases, unknown-roster exclusion, and preservation of native image quality.
- General inference now accepts native-quality v3-format checkpoints and their number-adapter
  weights; optional roster-number fusion is exposed. Actual latest/legacy loading, fresh saved
  embeddings, cached rankings and independent fused scores passed the integration check.
- Processes: persistent tmux `soccer_full_v8_clean` trains and scores; `soccer_full_v8_report`
  waits for the independent recount, then writes the run-specific completion report. Training
  log: `reports/soccernet_full_v2/training_v8_clean.log`. Results pending.

This entry supersedes any earlier description of auxiliary-pretrained runs as fully
player-disjoint. Their frozen supervised tables were disjoint; all soccer training inputs were not.

V8 first checkpoint: head epoch 1, whole-validation photo Rank-1 10.55%, mAP 15.47%; per-match fused validation Rank-1 33.75%, MRR 53.94% at photo weight 0.4. This is an early validation result, not a test result.

### R9 — completed — `soccernet_full_v8` — 2026-10-21

- **Manifest** `4f9de598127cd103` (unchanged)
- **Splits** train 265/9,140/2,123, val 115/1,043/324, test 182/1,448/818 (players/queries/references)
- **Evaluation** per-match roster (12 test matches) for photo and fused scores; whole-split photo
  retrieval reported alongside
- **Changed from previous** Auxiliary self-supervised stage removed (0 epochs) after the leakage
  audit; fresh pretrained CLIP, fresh projection and number adapter. Everything else as v7:
  number weight 2.0, flat-kit 0.2, unknown number uniform, fused validation selection, no added
  quality degradation. Early-stopping patience now follows the selected metric.
- **Training** head 29 epochs (last improvement 21), fine-tune 11 (early-stopped); fused selection
  chose fine-tune epoch 3, validation fused MRR 66.76%, Rank-1 52.35% at alpha 0.5; validation
  whole-split photo Rank-1 12.75%, mAP 16.05%. Photo rule would have chosen head epoch 6 (`best_photo.pt`,
  fused MRR 63.91%).
- **Result**, test, alpha chosen on validation:

| | v7 (leaky aux.) | **v8 (clean)** | v8 − v7, player-clustered 95% CI |
|---|---:|---:|---:|
| per crop Rank-1 | 41.09% | **43.44%** | +2.35 [-0.68, +5.27] |
| per crop MRR | 56.59% | **57.97%** | +1.38 [-0.95, +3.67] |
| per pool Rank-1 | 54.10% | **55.74%** | +1.64 [-4.40, +7.65] |
| per pool top-3 | 81.97% | 79.23% | -2.73 [-8.11, +2.20] |
| per pool MRR | 69.82% | **69.98%** | +0.16 [-3.63, +3.79] |
| photo per-match Rank-1 / mAP | 25.35 / 29.50% | 23.07 / 28.23% | -2.28 [-5.52, +0.92] / -1.27 [-3.10, +0.45] |
| photo whole-split Rank-1 / mAP | 10.29 / 10.34% | 7.80 / 8.96% | not bootstrapped |
| number read within roster | 31.4% | **33.8%** | — |

  Baselines unchanged: frozen CLIP per-match 14.85% / 19.12% (whole-split 3.87% / 4.44%), NBA
  projection 14.99% / 23.18% (2.35% / 4.44%). Independent verification passed.

- **Visualisation** none generated for v8; v7's `test_visualisation_fused_v7/` remains the latest sheet
- **Findings**
  - **The auxiliary stage was not doing the work.** Removing it, and with it every leaked
    held-out identity, leaves fused scores level with v7 or nominally higher; no difference is
    distinguishable. The leak did not inflate the headline fused numbers in any measurable way.
  - Photo-only retrieval is nominally lower without auxiliary pretraining (−2.3 Rank-1 per match,
    −2.5 whole-split) while number reading is higher, so the fused score shifts weight toward the
    number channel (validation alpha 0.5 vs 0.4). Neither photo difference is distinguishable.
  - Fused selection again beats photo selection inside the run (validation fused MRR 66.76% vs
    63.91%), consistent with R8.
- **Disposition** kept as current — the clean reference model; v7 is a historical diagnostic.
- **Artifacts** `runs/soccernet_full_v8/{best.pt,best_photo.pt,history.jsonl}`,
  `reports/soccernet_full_v2/{results_soccernet_full_v8.md,verification_soccernet_full_v8.json,number_channel_soccernet_full_v8_best.json}`,
  `reports/soccernet_full_v2/fused_comparison_soccernet_full_v8_vs_soccernet_full_v7.json`
  (new `experiments/soccernet_full_v2/compare_fused.py`), `reports/soccernet_full_v2/run_comparison_v8_vs_v7.json`.
  Shared archives: see `DATA.md`.
