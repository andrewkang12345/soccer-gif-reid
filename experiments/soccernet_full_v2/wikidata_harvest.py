"""Harvest public Wikidata footballer entities with a free photo, one SPARQL query per birth year.

The MediaWiki action API is aggressively rate limited for anonymous clients, so entity
resolution uses the query service instead: a small number of paced SPARQL requests replace
thousands of per-player API calls. Results are cached per query so the harvest is resumable.
"""
import json,time
from xml.etree import ElementTree
from pathlib import Path
import requests
from soccer_gif_reid.common import atomic_json
from .metadata import DATA

ENDPOINT='https://query.wikidata.org/sparql'
HEADERS={'User-Agent':'SoccerReIDResearch/0.1 (non-commercial player re-identification research)',
 'Accept':'application/sparql-results+json'}
CACHE=DATA/'wikidata_sparql'
YEARS=range(1979,2001)

IMAGES='''SELECT ?p ?name ?img ?cat WHERE {
 ?p wdt:P106 wd:Q937857 ; wdt:P18 ?img ; wdt:P569 ?dob ; rdfs:label ?name .
 FILTER(YEAR(?dob)=%d) FILTER(LANG(?name)="en")
 OPTIONAL { ?p wdt:P373 ?cat }
}'''
CLUBS='''SELECT ?p ?club WHERE {
 ?p wdt:P106 wd:Q937857 ; wdt:P18 ?i ; wdt:P569 ?dob ; wdt:P54 ?c .
 ?c rdfs:label ?club .
 FILTER(YEAR(?dob)=%d) FILTER(LANG(?club)="en")
}'''
ALIASES='''SELECT ?p ?alias WHERE {
 ?p wdt:P106 wd:Q937857 ; wdt:P18 ?i ; wdt:P569 ?dob ; skos:altLabel ?alias .
 FILTER(YEAR(?dob)=%d) FILTER(LANG(?alias) IN ("en","es","it","de","fr","pt","nl"))
}'''


NS='{http://www.w3.org/2005/sparql-results#}'


def parse(response,fmt):
 """The JSON writer occasionally emits raw control characters or truncates; XML survives both."""
 if fmt=='json':return json.loads(response.text,strict=False)['results']['bindings']
 rows=[]
 for result in ElementTree.fromstring(response.text).iter(NS+'result'):
  row={}
  for binding in result.findall(NS+'binding'):
   value=next(iter(binding),None)
   if value is not None and value.text is not None:row[binding.get('name')]={'value':value.text}
  if row:rows.append(row)
 return rows


def query(name,text,failures=None):
 CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/(name+'.json')
 if path.exists():
  try:return json.loads(path.read_text())
  except Exception:path.unlink()
 for attempt in range(6):
  fmt='json' if attempt<2 else 'xml'
  try:
   r=requests.get(ENDPOINT,params={'query':text,'format':fmt},
    headers=dict(HEADERS,Accept='application/sparql-results+json' if fmt=='json' else 'application/sparql-results+xml'),timeout=(10,300))
   if r.status_code in (429,500,502,503,504):
    time.sleep(min(180.,float(r.headers.get('Retry-After',20))*(attempt+1)));continue
   r.raise_for_status();rows=parse(r,fmt)
   path.write_text(json.dumps(rows));print(name,len(rows),fmt,flush=True);return rows
  except Exception as e:
   print(name,'retry',attempt,fmt,str(e).splitlines()[0],flush=True)
   time.sleep(20*(attempt+1))
 # A year that never parses is recorded and skipped: it weakens disambiguation, not correctness.
 if failures is not None:failures.append(name)
 print(name,'unavailable',flush=True);return []


def table(write=True):
 """Assemble the entity table from whatever cached year queries exist so far."""
 people={}
 for path in sorted(CACHE.glob('images_*.json')):
  for row in json.loads(path.read_text()):
   qid=row['p']['value'].rsplit('/',1)[-1]
   people.setdefault(qid,{'qid':qid,'name':row['name']['value'],'aliases':[],'clubs':[],
    'image':row['img']['value'],'commons_category':row.get('cat',{}).get('value'),
    'birth_year':int(path.stem.split('_')[-1])})
 for kind,field in [('clubs','club'),('aliases','alias')]:
  for path in sorted(CACHE.glob(kind+'_*.json')):
   for row in json.loads(path.read_text()):
    person=people.get(row['p']['value'].rsplit('/',1)[-1])
    if person and row[field]['value'] not in person[kind]:person[kind].append(row[field]['value'])
 out=sorted(people.values(),key=lambda r:r['qid'])
 if write:(DATA/'wikidata_footballers.json').write_text(json.dumps(out))
 return out


def main():
 people={};failures=[]
 for year in YEARS:
  for row in query(f'images_{year}',IMAGES%year,failures):
   qid=row['p']['value'].rsplit('/',1)[-1]
   people.setdefault(qid,{'qid':qid,'name':row['name']['value'],'aliases':[],'clubs':[],
    'image':row['img']['value'],'commons_category':row.get('cat',{}).get('value'),'birth_year':year})
  time.sleep(2)
 for year in YEARS:
  for row in query(f'clubs_{year}',CLUBS%year,failures):
   person=people.get(row['p']['value'].rsplit('/',1)[-1])
   if person and row['club']['value'] not in person['clubs']:person['clubs'].append(row['club']['value'])
  time.sleep(2)
 for year in YEARS:
  for row in query(f'aliases_{year}',ALIASES%year,failures):
   person=people.get(row['p']['value'].rsplit('/',1)[-1])
   if person and row['alias']['value'] not in person['aliases']:person['aliases'].append(row['alias']['value'])
  time.sleep(2)
 out=table();atomic_json(DATA/'wikidata_harvest_audit.json',{'entities':len(out),'unavailable_queries':failures,'years':[YEARS.start,YEARS.stop-1]})
 print(json.dumps({'entities':len(out),'with_commons_category':sum(1 for r in out if r['commons_category']),
  'with_clubs':sum(1 for r in out if r['clubs']),'with_aliases':sum(1 for r in out if r['aliases'])},indent=2),flush=True)


if __name__=='__main__':main()
