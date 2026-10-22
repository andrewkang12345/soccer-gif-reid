"""Conservative per-action team assignment using roster-exclusive jersey evidence."""
import json,hashlib,unicodedata,re
from collections import Counter,defaultdict
from pathlib import Path
from .metadata import DATA,ROOT

def main():
 mapped=[];decisions=[]
 norm=lambda s:re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s.lower()).encode('ascii','ignore').decode())
 old=json.loads((ROOT/'data/soccernet_named_v1/players.json').read_text())
 known={r['pid'] for r in old};known_names={norm(r['name']) for r in old}
 all_names={}
 for game in json.loads((DATA/'games.json').read_text()):
  cap=json.loads((DATA/'metadata'/game['key']/'Labels-caption.json').read_text())
  for side in ('home','away'):
   for person in cap['lineup'][side]['players']:
    all_names.setdefault(norm(person.get('long_name') or person['name']),set()).add(person['hash'])
 ambiguous={name for name,pids in all_names.items() if len(pids)>1}
 for r in map(json.loads,(ROOT/'data/arg_fra_expanded_gallery_v1/gallery.jsonl').read_text().splitlines()):known_names.add(norm(r['name']))
 def player_split(p):
  if norm(p.get('long_name') or p['name']) in ambiguous:return 'exclude'
  if p['hash'] in known or norm(p.get('long_name') or p['name']) in known_names:return 'train'
  bucket=int(hashlib.sha256(('player-v2:'+p['hash']).encode()).hexdigest()[:8],16)%10
  return 'test' if bucket==0 else 'val' if bucket==1 else 'train'

 for g in json.loads((DATA/'games.json').read_text()):
  folder=DATA/'metadata'/g['key'];caption=json.loads((folder/'Labels-caption.json').read_text());v3=json.loads((folder/'Labels-v3.json').read_text())
  roster={side:defaultdict(list) for side in ('home','away')}
  for side in roster:
   for p in caption['lineup'][side]['players']:
    if str(p.get('shirt_number','')).isdigit():roster[side][str(p['shirt_number'])].append(p)
  sets={s:set(x) for s,x in roster.items()}
  for action,ad in v3['actions'].items():
   frames=[(action,ad)]+[(f,v3['replays'][f]) for f in ad['linked_replays']]
   numbers={side:{str(b.get('ID')) for _,im in frames for b in im['bboxes'] if b['class'] in ['Player team '+side,'Goalkeeper team '+side] and str(b.get('ID')).isdigit()} for side in ('left','right')}
   choices=[]
   for left,right in [('home','away'),('away','home')]:
    orientation={'left':left,'right':right};contradictions=[];votes=set()
    for side,nums in numbers.items():
     own=orientation[side];other='away' if own=='home' else 'home'
     contradictions +=[(side,n) for n in nums if n not in sets[own]]
     votes|={(side,n) for n in nums if n in sets[own]-sets[other]}
    if not contradictions and len(votes)>=2:choices.append((orientation,sorted(votes)))
   accepted=len(choices)==1
   decisions.append({'game':g['game'],'action':action,'accepted':accepted,'number_sets':{k:sorted(x) for k,x in numbers.items()},'reason':'unique mapping with >=2 distinct roster-exclusive team/number witnesses and zero unmatched numbers' if accepted else 'ambiguous or inconsistent roster evidence','assignment':choices[0][0] if accepted else None,'witnesses':choices[0][1] if accepted else []})
   if not accepted:continue
   orientation=choices[0][0]
   for filename,im in frames:
    for b in im['bboxes']:
     cl=b['class'];number=str(b.get('ID'))
     if cl not in ['Player team left','Player team right','Goalkeeper team left','Goalkeeper team right'] or not number.isdigit():continue
     side=orientation[cl.split()[-1]];people=roster[side].get(number,[])
     if len(people)!=1:continue
     p=people[0]
     if player_split(p)!=g['split']:continue
     box=b['points'];h=box['y2']-box['y1'];w=box['x2']-box['x1']
     if h<48 or w<16:continue
     if g['split']!='train' and filename!=action:continue
     mapped.append(dict(pid=p['hash'],name=p.get('long_name') or p['name'],team=caption[side]['name'],number=int(number),game=g['game'],split=g['split'],action=action,frame=filename,view='broadcast' if filename==action else 'replay',uai=b['UAI'],height=h,width=w,label_source='SoccerNet-v3 numeric jersey + uniquely consistent per-action roster team mapping + SoccerNet-Caption player hash/name'))
 byid=defaultdict(list)
 for r in mapped:byid[r['pid']].append(r)
 # Exclude every real identity crossing custom split boundaries, never silently assign it.
 overlap={pid for pid,rs in byid.items() if len({r['split'] for r in rs})>1}
 mapped=[r for r in mapped if r['pid'] not in overlap]
 (DATA/'mapped_boxes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in mapped))
 (DATA/'mapping_decisions.json').write_text(json.dumps(decisions,indent=2))
 players=[]
 for pid,rs in byid.items():
  if pid in overlap:continue
  players.append({k:rs[0][k] for k in ('pid','name','team','split')}|{'crops':len(rs),'broadcast_crops':sum(r['view']=='broadcast' for r in rs)})
 players.sort(key=lambda r:(r['split'],-r['crops'],r['pid']))
 (DATA/'players.json').write_text(json.dumps(players,indent=2));print('Actions',Counter(d['accepted'] for d in decisions),'overlap excluded',overlap,'boxes',Counter(r['split'] for r in mapped),'players',Counter(r['split'] for r in players));print('Unique real names',len({r['name'] for r in players}))
if __name__=='__main__':main()
