"""Audit auxiliary self-supervised crops against evaluation roster identities."""
import json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
from soccer_gif_reid.common import atomic_json
from .metadata import DATA,ROOT

def main():
 rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
 held={r['pid']:r['split'] for r in rows if r['split']!='train'};names={r['pid']:r['name'] for r in rows}
 config=json.loads((ROOT/'runs/soccernet_full_v7/config.json').read_text())
 people=[json.loads(s) for s in (DATA/'ssl_pairs.jsonl').read_text().splitlines()][:config['ssl_persons']]
 decisions={(r['game'],r['action']):r for r in json.loads((DATA/'mapping_decisions.json').read_text()) if r['accepted']}
 wanted=defaultdict(set)
 for person in people:
  for view in person['views']:wanted[person['match']].add(Path(view['path']).stem.rsplit('-',2)[1])
 labels={};unknown_metadata=0
 for game,uais in wanted.items():
  folder=DATA/'metadata'/hashlib.sha256(game.encode()).hexdigest()[:12]
  if not (folder/'Labels-caption.json').exists() or not (folder/'Labels-v3.json').exists():continue
  cap=json.loads((folder/'Labels-caption.json').read_text());v3=json.loads((folder/'Labels-v3.json').read_text())
  roster={side:defaultdict(list) for side in ['home','away']}
  for side in roster:
   for player in cap['lineup'][side]['players']:roster[side][str(player.get('shirt_number'))].append(player)
  for action,ad in v3['actions'].items():
   assignment=decisions.get((game,action),{}).get('assignment')
   if not assignment:continue
   frames=[ad]+[v3['replays'][f] for f in ad['linked_replays']]
   for frame in frames:
    for box in frame['bboxes']:
     if box['UAI'] not in uais or not str(box.get('ID')).isdigit():continue
     side=box['class'].split()[-1]
     if side not in assignment or not box['class'].startswith(('Player ','Goalkeeper ')):continue
     players=roster[assignment[side]].get(str(box['ID']),[])
     if len(players)==1:labels[(game,box['UAI'])]=players[0]['hash']
 hits=[];resolved=0
 for person in people:
  game=person['match'];pids={labels[(game,Path(v['path']).stem.rsplit('-',2)[1])] for v in person['views'] if (game,Path(v['path']).stem.rsplit('-',2)[1]) in labels}
  if not (DATA/'metadata'/hashlib.sha256(game.encode()).hexdigest()[:12]/'Labels-caption.json').exists():unknown_metadata+=1
  if len(pids)==1:
   resolved+=1;pid=next(iter(pids))
   if pid in held:hits.append({'person':person['person'],'game':game,'pid':pid,'split':held[pid],'name':names[pid]})
 result={'ssl_groups_used':len(people),'groups_resolved_to_roster_player':resolved,'groups_without_available_roster_metadata':unknown_metadata,'groups_containing_heldout_player':len(hits),
  'heldout_identities_by_split':{s:len({r['pid'] for r in hits if r['split']==s}) for s in ['val','test']},
  'known_overlap_groups_by_split':dict(Counter(r['split'] for r in hits)),'evidence':hits,
  'note':'Numeric jersey and an accepted action team assignment identify evaluation players in different self-supervised training matches. Unresolved anonymous groups cannot certify identity separation.'}
 atomic_json(ROOT/'reports/soccernet_full_v2/ssl_identity_audit.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2),flush=True)
if __name__=='__main__':main()
