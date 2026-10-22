"""Reconcile extracted/masked rows with the final frozen mapping revision."""
import json
from collections import Counter
from .metadata import DATA

def main():
 mapping={(r['game'],r['uai']):r for r in map(json.loads,(DATA/'mapped_boxes.jsonl').read_text().splitlines())}
 for name in ['queries.jsonl','masked_queries.jsonl']:
  p=DATA/name
  if not p.exists():continue
  rows=[json.loads(s) for s in p.read_text().splitlines()];result=[]
  for r in rows:
   key=(r['game'],r['uai'])
   if key not in mapping:continue
   label=mapping[key];assert r['pid']==label['pid']
   result.append(dict(r,**{k:label[k] for k in ('name','team','split','number','label_source')}))
  p.write_text(''.join(json.dumps(r)+'\n' for r in result));print(name,Counter(r['split'] for r in result),flush=True)
if __name__=='__main__':main()
