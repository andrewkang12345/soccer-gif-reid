"""Freeze all screened references and disjoint query splits after documented review."""
import json,hashlib,unicodedata,re
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
from PIL import Image
from soccer_gif_reid.common import atomic_json,digest_file,read_rgb,write_manifest
from .metadata import DATA,ROOT

MIN_REFERENCES=3
MIN_EVALUATION_REFERENCES=1


def evidence_key(row):
 """How this reference's identity was established, for the frozen audit."""
 if row.get('reference_type')=='wikidata_entity':return 'wikidata_entity/'+str(row.get('image_role'))
 return 'named_source/'+str(row.get('reference_type'))
def normalized(s):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s.lower()).encode('ascii','ignore').decode())
def main():
 ready=json.loads((DATA/'page_search_complete.json').read_text())
 assert ready['players_searched']==len(json.loads((DATA/'selected_players.json').read_text()))
 review=json.loads((DATA/'manual_reference_review.json').read_text())
 assert review['complete'] is True
 assert review['screened_manifest_sha256']==digest_file(DATA/'screened_gallery.jsonl')
 label=json.loads((ROOT/'reports/soccernet_full_v2/label_review/automated_audit.json').read_text())
 assert label['visual_review_complete'] and not label['roster_join_failures']
 players={r['pid']:r for r in json.loads((DATA/'players.json').read_text())}
 gallery=[json.loads(s) for s in (DATA/'screened_gallery.jsonl').read_text().splitlines()]
 query=[json.loads(s) for s in (DATA/'masked_queries.jsonl').read_text().splitlines()]
 counts=Counter(r['pid'] for r in gallery)
 # Training wants several references per identity; evaluation should look like deployment, where
 # a roster gives whatever photographs exist — often one. Holding evaluation to three references
 # both shrank the test set to four matches and flattered it with unusually good galleries.
 split_of={r['pid']:r['split'] for r in query}
 eligible={pid for pid,n in counts.items() if pid in split_of
           and n>=(MIN_REFERENCES if split_of[pid]=='train' else MIN_EVALUATION_REFERENCES)}
 # Copy the downloaded source bytes unchanged: no local crop, resize or re-encode of references.
 gallery=[r for r in gallery if r['pid'] in eligible];query=[r for r in query if r['pid'] in eligible]
 sha_pids=defaultdict(set)
 for r in gallery:sha_pids[r['sha256']].add(r['pid'])
 assert all(len(pids)==1 for pids in sha_pids.values())
 for r in gallery:
  p=players[r['pid']];old=Path(r['path'])
  dest=DATA/'gallery'/p['split']/(p['name'].replace(' ','_')+'_'+p['pid'])/(r['candidate_id'].split('_')[-1]+'.png')
  dest.parent.mkdir(parents=True,exist_ok=True)
  box=tuple(int(round(v)) for v in r['person_screen']['crop_box'])
  # Written losslessly from the decoded source pixels: the gallery holds the dominant person,
  # while the untouched source file and its hash stay recorded for provenance.
  if not dest.exists():read_rgb(old).crop(box).save(dest,'PNG',compress_level=4)
  r['gallery_sha256']=digest_file(dest);r['crop_box']=list(box)
  evidence='Unique Wikidata football-player entity link ('+str(r.get('wikidata_qid'))+', '+str(r.get('image_role'))+')' if r.get('reference_type')=='wikidata_entity' else 'Explicit player name in the public source title/URL'
  r.update(candidate_path=str(old),path=str(dest),split=p['split'],domain='reference',label_source=evidence+' plus automated dominant-person and duplicate screening; manually sampled composition review, not independently certified identity')
 allrows=query+gallery
 for key in ('pid','sha256'):
  sets={s:{r[key] for r in allrows if r['split']==s} for s in ('train','val','test')}
  for a,b in [('train','val'),('train','test'),('val','test')]:assert not sets[a]&sets[b],(key,a,b)
 names={s:{normalized(r['name']) for r in allrows if r['split']==s} for s in ('train','val','test')}
 games={s:{r['game'] for r in query if r['split']==s} for s in ('train','val','test')}
 for a,b in [('train','val'),('train','test'),('val','test')]:
  assert not names[a]&names[b],('normalized name',a,b)
  assert not games[a]&games[b],('game',a,b)
 assert not {r['sha256'] for r in query}&{r['sha256'] for r in gallery}
 assert len({r['sha256'] for r in gallery})==len(gallery)
 assert len({r['gallery_sha256'] for r in gallery})==len(gallery)
 splits={s:{'players':len({r['pid'] for r in query if r['split']==s}),'queries':sum(r['split']==s for r in query),'references':sum(r['split']==s for r in gallery),'games':len(games[s])} for s in ('train','val','test')}
 assert all(splits[s]['players']>=10 and splits[s]['queries']>=30 for s in ('val','test')),'Too few reliably paired evaluation players'
 for filename,rs in [('gallery.jsonl',gallery),('final_queries.jsonl',query),('manifest.jsonl',allrows)]:write_manifest(DATA/filename,rs)
 audit={'review_passed':True,'frozen_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest_file(DATA/'manifest.jsonl'),
  'mapping_sha256':digest_file(DATA/'mapped_boxes.jsonl'),'gallery_manifest_sha256':digest_file(DATA/'gallery.jsonl'),
  'query_manifest_sha256':digest_file(DATA/'final_queries.jsonl'),'splits':splits,'reference_count_distribution':dict(Counter(counts[pid] for pid in eligible)),
  'candidate_players':len(players),'excluded_players':[dict(p,usable_references=counts[pid]) for pid,p in players.items() if pid not in eligible],
  'reference_cap':None,'minimum_distinct_references':{'train':MIN_REFERENCES,'val':MIN_EVALUATION_REFERENCES,'test':MIN_EVALUATION_REFERENCES},
  'gallery_region':'Padded dominant-person crop of each source photograph, written losslessly; source file, source hash and crop box are recorded per reference.',
  'reference_evidence':dict(sorted(Counter(evidence_key(r) for r in gallery).items())),'label_audit_sample':label['sample'],'reference_review':review,
  'disjointness':'Real roster player hash, normalized name, source game and original image hashes. No exact query/reference duplicates.',
  'caveats':['Full public named, reliably paired subset; anonymous or ambiguous SoccerNet data excluded.',
   'Automatic roster and source-caption labels remain weak supervision; sampled review cannot certify all identities.',
   'Reference collection uses public web search, named public pages and Wikidata/Commons entity links, not Google-only.',
   'Entity-linked references carry a recorded Wikidata QID instead of source-title name evidence; Commons person categories are volunteer curated and can still contain a misfiled or group photograph.',
   'No added blur, downsampling, JPEG/noise degradation. Kit masks are conservative geometry, not perfect parsing.',
   'References are detector crops, so inference must crop roster photographs the same way; a detector miss changes the stored region.',
   'Prior pilot and Argentina-France player identities are excluded from new validation/test; Argentina-France is development-only.',
   'CLIP ViT-L/14 pretraining plausibly saw these public reference photos with the player name attached, so absolute retrieval numbers may not transfer to a roster of photographs the backbone has never seen.'],
  'protocol_sha256':digest_file(ROOT/'reports/soccernet_full_v2/protocol.md')}
 atomic_json(DATA/'final_audit.json',audit)
 print(json.dumps({'splits':splits,'reference_counts':audit['reference_count_distribution'],'excluded_players':len(audit['excluded_players'])},indent=2),flush=True)
if __name__=='__main__':main()
