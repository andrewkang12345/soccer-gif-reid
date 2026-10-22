# SoccerNet full named subset v2 - result

Experiment `experiments/soccernet_full_v2`; data `data/soccernet_full_v2`; reports `reports/soccernet_full_v2`.
Manifest `5abe05e534027fb5c1b0da9193c53a67194ccf65632358cd9289161a1524bb5b`, frozen 2026-09-15T04:57:39.757284+00:00.

## Frozen dataset

| split | players | queries | references | games |
|---|---:|---:|---:|---:|
| train | 219 | 9715 | 1703 | 281 |
| val | 60 | 343 | 383 | 9 |
| test | 33 | 153 | 299 | 26 |

1741 roster players were mapped and segmented; 1429 are excluded for
having fewer than 3 distinct screened references. Reference evidence in the frozen
gallery: `{"named_source/fullbody_search": 240, "named_source/named_web_page": 1346, "named_source/portrait_search": 732, "wikidata_entity/wikidata_image": 67}`.

## What blocked the run, and what unblocked it

The preparation the previous session left running could not have been frozen. Screening the
8,695 collected photographs kept 2,321 over 634 players, and only 223 players reached three
distinct references - three of them in validation, against a freeze gate of ten players and
thirty queries. The cause is in `page_search_complete.json`: searches were issued in batches of
four players and each saved record holds the merged result set, so `collect_pages`, which
correctly keeps only pages naming the player, discards three quarters of every batch.

Four changes were made, each recorded in `protocol.md` with its cost:

1. **Entity-linked references.** A roster name is linked to a Wikidata association-football-player
   entity only when exactly one entity born 1979-2000 with a free photo carries that normalized
   name, or when club history leaves exactly one candidate. 1498 of 1741 players link,
   29 are dropped as ambiguous. References are the entity's `image` claim and its Commons
   person category, ranked by a title heuristic. Entity and category lookups run through the
   public SPARQL query service, including its `wikibase:mwapi` bridge to Commons, because the
   anonymous MediaWiki action API budget is far too small for this many lookups.
2. **Gallery region.** References are stored as the padded dominant-person crop of the source
   photograph rather than the whole image. Every query in this dataset is itself a tight player
   box about 60 pixels wide, so the old rule - main person must fill a fifth of the frame - was
   rejecting ordinary press pictures on composition alone.
3. **Name evidence.** A page's own hero image is now accepted when the page *address* names the
   player, not only when the image URL does; and club-name tokens, a host label equal to the club
   name, or a host label spelled as a subsequence of it count as football context, which stops
   official club sites (mancity.com, bayer04.de, herthabsc.com, psv.nl) from being rejected for
   using an abbreviation the roster never uses. Each reference records which evidence kept it.
4. **Validation re-cut games-first** (`repartition.py`, approved by the user). Validation games are
   chosen before players, every player in them becomes a validation player, and those players'
   crops in all other games are dropped: 3749 crops. Player, normalized-name and game
   disjointness are unchanged and still asserted at freeze time; the test split is untouched.

Screening now keeps 3134 references over 892 players, 316 of them with three or more.

## What did not work

Wikimedia's content host answered roughly 35-40 images an hour from this address after an initial
burst, with repeated 600-second cooldowns, so only 197 of the 10219 entity-linked candidates were
fetched - and entity-linked references were visibly the cleanest class in review. The remaining
candidates stay queued in `wiki_candidates.jsonl` with their entity links, so a later collection
run needs no new search or linkage work. Reference counts per player in this dataset are a
collection limit, not a property of the identities.

## Label review

Two rounds of visual inspection over contact sheets of the stored crops, stratified by name
evidence: 270 references across three evidence classes, then 360 across four. 35 candidates are
excluded by ID. The second round found 4 wrong identities in 360 (about 1 percent) - a club
manager shown on an article about a player twice, a camera operator, and Granit Xhaka shown for
his brother Taulant - and 6 composition or quality defects. No defect was found in the 90
entity-linked references examined. This remains weak supervision; a sampled visual check cannot
certify identities.

## Training

Two stages: the 790,272-parameter GIF projection head on frozen CLIP ViT-L/14, then the final two
CLIP blocks at a lower backbone learning rate. 500 identity-balanced batches of 16 per epoch,
30 head epochs and 15 fine-tuning epochs, validation-mAP selection with patience 8. Query
augmentation randomizes kit, background and modest geometry only - no blur, downsampling, JPEG
or noise. Started 343-query validation at Rank-1 13.4%, mAP 9.1% with the untrained head over
60 identities. Test is scored once, after both stages, alongside frozen CLIP and the NBA-trained
projection on the identical held-out split.

Run: `tmux attach -t soccer_reid_train`; log `reports/soccernet_full_v2/training.log`; resume with
`bash experiments/soccernet_full_v2/run_training.sh`.

## Result

Selected by validation mAP alone: fine-tuning stage, epoch 5 (validation Rank-1 22.74%, mAP 21.93%).
Scored once on the untouched test split of 33 players, 153 queries and 299 reference images.

| model | Rank-1 | Rank-5 | mAP |
|---|---:|---:|---:|
| soccer_full (this run) | 35.29% | 58.82% | 33.63% |
| frozen CLIP ViT-L/14 | 26.80% | 49.67% | 19.05% |
| NBA-trained projection | 11.11% | 21.57% | 11.42% |

Player-cluster bootstrap mAP difference against frozen CLIP: +14.58 points,
95% interval [7.18, 22.87]. Against the NBA-trained projection:
+22.21 points, 95% interval [14.23, 30.72]. Both intervals exclude zero.
Test metrics were recounted independently from the saved embeddings and matched to 1e-6.

Three things in this table matter more than the headline number.

**The NBA-trained projection is worse than no projection at all** - 11.42% mAP against frozen
CLIP's 19.05%. A head trained on MuPNIT-ReID does not merely fail to transfer to soccer, it
destroys usable CLIP structure. Soccer training is not an optional refinement here.

**Validation and test are not comparable to each other.** Validation ranks against 60 identities
and 383 reference images, test against 33 and 299, so test mAP is higher for the same model.
Only the within-split comparisons above carry information.

**The head saturates almost immediately and the backbone does not.** The frozen-backbone stage
peaked at epoch 8 (20.44% validation mAP) while training loss kept falling from 46.7 to 9.1;
unfreezing the last two CLIP blocks moved validation mAP from 20.44% to 21.93%. Training loss
falling by a factor of five while validation mAP stays flat is the head memorising 219
identities, not learning a transferable one - an argument for more identities rather than
longer training or a bigger head.

## What this run does not establish

- Reference counts per player are a collection limit, not a property of the identities: 8,700
  entity-linked candidates were found and could not be fetched. The most direct next experiment
  is to collect them and retrain, not to change the model.
- CLIP ViT-L/14 pretraining plausibly saw these public photographs with the player's name
  attached. Absolute numbers may not transfer to a roster of photographs the backbone has never
  seen.
- Roster and source-context labels are automatic weak supervision with about 1 percent wrong
  identity in the reviewed sample; the test split carries the same label process as training.
- 33 test identities is a small gallery. Rank-1 of 35% there is not a claim about a full
  matchday roster of 30+ players from two teams under broadcast conditions.
