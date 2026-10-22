import json,requests,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'data/soccernet_named_v1'
# Different clubs and matches across splits, plus a global real-player overlap audit.
CHOICES={'train':['Chelsea 1 - 1 Burnley','Barcelona 3 - 1 Paris SG','Real Madrid 1 - 0 Liverpool','Bayern Munich 2 - 0 AS Roma','Paris SG 1 - 1 Chelsea','Paris SG 1 - 3 Barcelona','Barcelona 2 - 0 Paris SG','Barcelona 3 - 0 Bayern Munich','Bayern Munich 3 - 2 Barcelona'],'val':['Dortmund 0 - 3 Juventus','Juventus 3 - 2 Fiorentina','Torino 2 - 1 Juventus'],'test':['Arsenal 1 - 3 Monaco','Crystal Palace 1 - 2 Arsenal','Manchester United 1 - 1 Arsenal']}
def main():
 games=json.loads((DATA/'available_games.json').read_text());selected=[]
 for split,patterns in CHOICES.items():
  for pattern in patterns:
   found=[g for g in games if '/2014-2015/' in g and g.endswith(pattern)];assert len(found)==1,found
   selected.append({'game':found[0],'split':split,'key':hashlib.sha256(found[0].encode()).hexdigest()[:12]})
 def fetch(item):
  d=DATA/'metadata'/item['key'];d.mkdir(parents=True,exist_ok=True)
  for f in ['Labels-caption.json','Labels-v3.json']:
   p=d/f
   if not p.exists():
    r=requests.get('https://huggingface.co/datasets/SoccerNet/SN-Labels/resolve/main/'+quote(item['game']+'/'+f,safe='/'),timeout=40);r.raise_for_status();p.write_bytes(r.content)
  return item
 with ThreadPoolExecutor(max_workers=6) as pool:selected=list(pool.map(fetch,selected))
 (DATA/'games.json').write_text(json.dumps(selected,indent=2))
 print(selected,flush=True)
if __name__=='__main__':main()
