"""Build cross-view player pairs from SoccerNet's own person links, for self-supervised training.

SoccerNet's ReID archives carry `person_uid`: the same athlete annotated in an action frame and
in its linked replays. That is a positive pair from a different camera at a different instant,
taken from annotation rather than from a tracker, so the identity switch that would poison a
tracklet-mined pair cannot happen. 62,714 persons in the training archive have two or more views,
and only a third of those crops carry a readable shirt number, so the great majority of this is
invisible to the labelled pipeline.

Two safeguards. Matches used by the frozen validation or test splits are excluded outright, so
self-supervised pretraining cannot see an evaluation match. And `person_uid` is scoped to one
action, so the same athlete in two actions counts as two identities: positives are clean,
negatives are noisy in the usual self-supervised way.
"""
import json,zipfile
from collections import defaultdict
from pathlib import Path
from soccer_gif_reid.common import atomic_json,write_manifest
from .metadata import DATA,ROOT

ARCHIVES=ROOT/'data/soccernet_named_v1/archives'
OUT=DATA/'ssl_crops'


def held_out_matches():
 """Matches any frozen evaluation split draws from, which pretraining must not touch."""
 rows=[json.loads(s) for s in (DATA/'final_queries.jsonl').read_text().splitlines()]
 return {r['game'] for r in rows if r['split'] in ('val','test')}


def groups(archive,blocked,minimum_views=2,min_side=24):
 """Persons in this archive with at least two views, outside the held-out matches."""
 with zipfile.ZipFile(archive) as z:
  name=next(x for x in z.namelist() if x.endswith('bbox_info.json'))
  info=json.loads(z.read(name))
 # train.zip stores one flat bbox table; valid/test nest it under "query" and "gallery".
 flat={}
 for key,value in info.items():
  if isinstance(value,dict) and 'height' in value:flat[key]=value
  elif isinstance(value,dict):flat.update({f'{key}/{k}':v for k,v in value.items()})
 people=defaultdict(list)
 for key,v in flat.items():
  if min(v['height'],v['width'])<min_side:continue
  match=v['relative_path'].rsplit('/',1)[0]
  if match in blocked:continue
  people[(match,v['relative_path'],v['action_idx'],v['person_uid'])].append(dict(v,key=key))
 return {k:v for k,v in people.items() if len({x['frame_idx'] for x in v})>=minimum_views}


def member(entry,root):
 """Rebuild the archive member path the ReID zips use for one bounding box."""
 return (f"{root}/{entry['relative_path']}/{entry['bbox_idx']}-{entry['action_idx']}-"
         f"{entry['person_uid']}-{entry['frame_idx']}-{entry['clazz']}-{entry['id']}-"
         f"{entry['UAI']}-{entry['height']}x{entry['width']}.png")


def main(limit_persons=0):
 blocked=held_out_matches();OUT.mkdir(parents=True,exist_ok=True)
 manifest=[];audit={}
 for archive in ('train.zip','valid.zip'):
  path=ARCHIVES/archive
  if not path.exists():continue
  root=archive.split('.')[0].replace('valid','valid')
  people=groups(path,blocked)
  index={}
  with zipfile.ZipFile(path) as z:
   names={x.rsplit('/',1)[-1]:x for x in z.namelist() if x.endswith('.png')}
   written=0
   for pid,(key,entries) in enumerate(sorted(people.items())):
    if limit_persons and pid>=limit_persons:break
    folder=OUT/f'{archive[:-4]}_{pid:06d}';saved=[]
    for e in entries:
     stem=member(e,root).rsplit('/',1)[-1]
     source=names.get(stem)
     if not source:continue
     folder.mkdir(parents=True,exist_ok=True)
     target=folder/stem
     if not target.exists():target.write_bytes(z.read(source))
     saved.append({'path':str(target),'frame_idx':e['frame_idx'],'number':e['id'],
                   'team':e['clazz'],'height':e['height'],'width':e['width']})
    if len({s['frame_idx'] for s in saved})>=2:
     manifest.append({'person':f'{archive[:-4]}_{pid:06d}','match':key[0],'action':key[2],
                      'views':saved});written+=len(saved)
    if pid%2000==0:print(archive,pid,len(people),'persons kept',len(manifest),'crops',written,flush=True)
  audit[archive]={'persons_with_two_views':len(people),'persons_written':len(manifest),'crops':written}
 write_manifest(DATA/'ssl_pairs.jsonl',manifest)
 audit['held_out_matches_excluded']=len(blocked)
 audit['total_persons']=len(manifest);audit['total_crops']=sum(len(m['views']) for m in manifest)
 audit['note']=('Positives are the same annotated person across linked views of one action. '
  'person_uid is action-scoped, so the same athlete in two actions is two identities here.')
 atomic_json(DATA/'ssl_pairs_audit.json',audit)
 print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
