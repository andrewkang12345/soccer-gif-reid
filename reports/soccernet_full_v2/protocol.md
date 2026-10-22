# Full reliably mapped SoccerNet experiment v2

Scope: all 372 available matches with both SoccerNet-v3 and caption metadata. Numeric jersey annotations are joined to per-match rosters only when one team-side assignment has at least two distinct roster-exclusive witnesses and zero contradictory numbers. Ambiguous short names shared by different roster hashes are excluded.

Splits are frozen by independent SHA256 game/player partitions (80/10/10). A crop is retained only when its player's partition agrees with its game's partition. All player identities/names exposed in the prior mapped pilot and Argentina-France are forced to the training partition. Validation/test use broadcast stills only. This discards otherwise mapped data to ensure both player and source-game separation; it does not mean every original SoccerNet crop is usable for this task.

Audit discovered that caption `teams` list order is not always home/away order. Team display/search labels now use explicit `home.name` and `away.name`; crop-to-person joins already use the correct home/away roster dictionaries. Prior pilot artifacts remain unchanged.

Reference collection: retain all screened distinct photos, with no eight-photo cap. Reject unverified name associations, group shots without a dominant subject, likely game/render assets, duplicates and query/reference overlaps. Image search hit a rate limit. Collection now also uses ordinary web search and named public photo/news pages, retaining image-level source provenance. Collection and final review are not yet complete.

Planned training: frozen CLIP ViT-L/14 projection-head stage with existing kit-invariance augmentation; then fine-tune the final CLIP blocks with a smaller learning rate, accepting only validation improvements. No additional blur/downsampling/JPEG degradation. Independent test scores are withheld until model selection is complete. No full-data training result is claimed yet.

Reference-review findings: sampled public pages sometimes associate a player's headline with another person in the featured image, and namesakes can appear on non-football websites. Featured-image candidates now require name evidence in the image URL, and named-page candidates require football/club source context. These are conservative heuristics, not calibrated identity guarantees. Image-specific captions can themselves be wrong. Manual exclusions are recorded by stable candidate IDs; the final report must disclose residual weak-label risk.

Verification: head/final-block gradient paths passed; interrupted-versus-continuous training matched within 7.5e-9 maximum parameter difference on a tiny prior-pilot-only integration check. Cached and uncached inference rankings matched exactly on prior-pilot training data. No new full-v2 test scoring occurred during these checks.

Entity-linked reference source (added after the first full screening pass). Web-page collection
alone screened down to 2,321 photos over 634 players, and only 223 players reached the three
distinct references required for eligibility — 3 of them in validation — so the frozen splits
could not have been evaluated. Reference collection was therefore extended with public
Wikimedia sources. A roster name is linked to a Wikidata entity only when exactly one
association-football-player entity born 1979-2000 with a free photo carries that normalized
name, or when club history (P54) leaves exactly one candidate; every other name is dropped as
ambiguous. For a linked player the references are that entity's own `image` (P18) claim and the
files in its Commons person category (P373), capped at 40 files per category. The identity
evidence for these photos is the recorded entity link (QID, matched label, match rule) rather
than a page title, so they bypass the source-title name test and are still subject to the
game-asset, nonphotographic, dominant-person, cross-identity and duplicate screens.

Limitations of that source: a Commons person category is curated by volunteers and can contain
group photographs or a misfiled image, so category membership is weaker evidence than the P18
claim; both are recorded per image. Wikimedia reference bytes are the original uploaded file, fetched from
the Commons content host, so every reference in this dataset keeps its original source bytes. The anonymous MediaWiki action API is too heavily rate limited for
this many lookups, so entity and category listing run through the public SPARQL query service
(`query.wikidata.org`, including its `wikibase:mwapi` bridge to Commons); only public read-only
endpoints are used.

Gallery region (changed after the first full screening pass). References are now stored as the
padded dominant-person crop of the source photograph rather than the whole image. The previous
rule required the main person to fill at least a fifth of the frame and rejected 1,886 otherwise
name-verified photographs purely on composition; those are ordinary press pictures in which the
player is alone but small. Cropping keeps them and also removes a domain gap, since every query
is itself a tight player box — the broadcast crops in this dataset are around 60 pixels wide, so
the retained-subject floor is 56 pixels on the short side with a detection confidence of at least
0.30 and no second person above 35% of the main area. Detector measurements are cached and the
thresholds are applied on top, so the recorded decision can be re-derived. Each reference records
its source file, source hash, crop box and the hash of the written crop. Inference crops roster
photographs with the same detector and thresholds by default (`--no-roster-crop` disables it) and
reports any roster photograph with no single dominant person. A detector miss changes the stored
region, so this is a systematic dependency of the gallery, not a neutral preprocessing step.

Validation re-cut (changed after the first full screening pass). The original split hashed players
and games independently and kept a crop only when both partitions agreed. That is strictly
disjoint but discards roughly nine tenths of every evaluation player's crops: validation held 35
players and 139 crops, and after reference screening only 3 of those players had the three
distinct references the freeze requires, so no checkpoint could have been selected. Validation is
now cut games-first: validation games are chosen before players, every player appearing in them
becomes a validation player, and those players' crops in all other games are dropped. Player,
normalized-name and game disjointness between splits are unchanged and still asserted at freeze
time. The test split is left exactly as the earlier partition produced it, and the identities
exposed by the earlier pilot and by the Argentina-France development clip remain in training and
cannot enter validation. The crops discarded by this re-cut are recorded in
`repartition_audit.json`.

Reference collection was capped by the public image hosts, not by the pipeline. Wikimedia's
content host answered roughly 40 images an hour from this address after an initial burst, with
600-second cooldowns, so only a small part of the 8,942 entity-linked candidates could be
fetched; the remainder stay queued in `wiki_candidates.jsonl` and can be collected later without
repeating the search or linkage work. The frozen dataset therefore under-represents what the
entity linkage found, and reference counts per player are a collection limit rather than a
property of the identities.

Name evidence (widened after the first full screening pass, with the reason recorded per
reference). Two screens were rejecting far more than the label noise they were built to stop.

The featured-image screen required the player's name inside the image URL, which official club
and newspaper sites rarely provide because their image filenames are hashes; it rejected 1,705
photographs. A page's own hero image is now also accepted when the page *address* names the
player, for example `/es-ES/futbol/primer-equipo-masculino/plantilla/sami-khedira` or
`/football/milan/andrea-poli-12494/`. That is far more specific than the headline match which
produced the wrong associations found in the earlier manual sample, because an article about
one player can carry another player's name in its title but not in its URL path. It recovers
about 1,283 photographs over 694 players.

The football-context screen required a keyword or the roster's own spelling of the club, and so
rejected official club sites whose domain is an abbreviation the roster never uses: mancity.com,
bayer04.de, herthabsc.com, psv.nl, atleticodemadrid.com. Club-name tokens, a host label equal to
the club name, and a host label spelled as a subsequence of it now count as club context, which
recovers about 522 photographs over 93 players. Entertainment and cinema sites that share a
player's name — the actual namesake risk, and the largest single rejected host — still fail every
test.

Both are recall relaxations on automatic weak supervision, so each retained reference records
which evidence kept it (`image URL names the player`, `page address names the player`,
`image caption or source title names the player`, `wikidata entity link`), the counts appear in
the screening audit and the freeze audit, and the manual review sample is drawn after these
rules so the residual error rate is judged on the gallery that is actually used.


Match-level evaluation (changed after the first result). Both evaluation splits are now cut
games-first, test before validation, and the test split is scored the way the model is used: the
gallery is the roster of the one match a crop came from. The earlier test split kept the original
independent player/game hash partition, which scattered 33 players over 27 games — one to five
players per match — so a match-level score was impossible and the only available number ranked
each crop against every test identity at once. Test now holds 61 players over 4 matches with
[21, 20, 20, 17] players in them.

This changes the difficulty in both directions and neither dominates. Every candidate in a match
wears one of two kits, so kit colour — which whole-split retrieval lets a model exploit — carries
almost no information; and the roster bounds the answer to about twenty people instead of the
whole split. Both numbers are reported, per-match first, with a per-match breakdown, because they
answer different questions.

The cost: the test split is no longer the untouched original partition. It is cut by the same
greedy rule as validation, which prefers matches whose players have reference coverage, so test
players are better photographed than a random SoccerNet player. Pilot and Argentina-France
identities are still excluded from both evaluation splits, and player, normalized-name and game
disjointness are still asserted at freeze time. Dropped by the re-cut: {"test player outside its test games": 1558, "val player outside its val games": 2460, "training player inside a val game": 170, "training player inside a test game": 44}.

Test-time augmentation is not used. Query augmentation is a training-time device; applying it at
evaluation would only be worth doing if it measurably improved test performance, and that has not
been tested.
