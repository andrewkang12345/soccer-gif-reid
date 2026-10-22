"""List Wikimedia Commons person-category files through the query service's MediaWiki API bridge.

The anonymous Commons `api.php` budget is far too small for a thousand categories, so the
listing is federated through query.wikidata.org (`wikibase:mwapi`), which performs the
category lookups server side. Only public read-only listing is used.
"""
import hashlib,json,time
import requests
from .metadata import DATA
from .wikidata_harvest import ENDPOINT,HEADERS

CACHE=DATA/'commons_sparql'
TEMPLATE='''PREFIX mwapi: <https://www.mediawiki.org/ontology#API/>
SELECT ?cat ?title WHERE {
 VALUES ?cat { %s }
 SERVICE wikibase:mwapi {
  bd:serviceParam wikibase:endpoint "commons.wikimedia.org" ;
   wikibase:api "Generator" ; mwapi:generator "categorymembers" ;
   mwapi:gcmtitle ?cat ; mwapi:gcmtype "file" ; mwapi:gcmlimit "%d" .
  ?title wikibase:apiOutput mwapi:title .
 }
}'''


def fetch(chunk,limit):
 """A query that exceeds the service time limit returns truncated JSON, so split and retry."""
 values=' '.join(json.dumps('Category:'+c) for c in chunk)
 for attempt in range(4):
  try:
   r=requests.get(ENDPOINT,params={'query':TEMPLATE%(values,limit),'format':'json'},headers=HEADERS,timeout=(10,300))
   if r.status_code in (429,500,502,503,504):
    time.sleep(min(120.,float(r.headers.get('Retry-After',15))*(attempt+1)));continue
   r.raise_for_status()
   return [[b['cat']['value'],b['title']['value']] for b in r.json()['results']['bindings']]
  except Exception:
   if len(chunk)>1 and attempt>=1:
    half=len(chunk)//2
    return fetch(chunk[:half],limit)+fetch(chunk[half:],limit)
   time.sleep(10*(attempt+1))
 raise RuntimeError('Commons category listing unavailable through the query service')


def listing(categories,limit=40,batch=12):
 """Map Commons category name -> [file titles]; cached per batch and resumable."""
 CACHE.mkdir(parents=True,exist_ok=True);out={};ordered=sorted(set(categories))
 for i in range(0,len(ordered),batch):
  chunk=ordered[i:i+batch]
  key=hashlib.sha256(('\n'.join(chunk)+f'|{limit}').encode()).hexdigest()[:20]
  path=CACHE/(key+'.json')
  if path.exists():rows=json.loads(path.read_text())
  else:
   rows=fetch(chunk,limit);path.write_text(json.dumps(rows));time.sleep(1)
  for cat,title in rows:out.setdefault(cat.split('Category:',1)[-1],[]).append(title)
  if (i//batch)%10==0:print('commons listing',min(i+batch,len(ordered)),len(ordered),'files',sum(len(v) for v in out.values()),flush=True)
 return out
