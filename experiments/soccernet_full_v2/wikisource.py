"""Resolve roster names to Wikidata footballer entities and list their public photos.

Entity linkage is the identity evidence for these references: a unique association-football
player entity whose label/alias matches the roster name, disambiguated by club history when
several candidates exist. Ambiguous names are dropped rather than guessed. Images come from
the entity's own `image` (P18) claim, its Commons person category (P373) and the English
Wikipedia lead image; Commons categories are curated per person, but can still contain group
photographs, so the existing dominant-person and duplicate screens still apply.
"""
import hashlib,json,re,time,threading,unicodedata
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import requests
from .metadata import DATA

UA='SoccerReIDResearch/0.1 (non-commercial player re-identification research)'
FOOTBALLER='Q937857'
CACHE=DATA/'wiki_cache'
LOCK=threading.Lock()
THROTTLE={'next':0.}
SESSION=threading.local()
ABBREV={'utd':'united','atl':'atletico','ath':'athletic','dep':'deportivo','mgladbach':'monchengladbach',
 'st':'saint','gladbach':'monchengladbach','psg':'parissaintgermain','sp':'sporting','dyn':'dynamo',
 'd':'dinamo','b':'borussia','w':'werder','m':'manchester','man':'manchester','fk':'','fc':'','cf':'','sv':'','ac':'','as':''}

def norm(s):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode())
def tokens(s):return [t for t in re.split(r'[^a-z0-9]+',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode()) if t]

def session():
 if not hasattr(SESSION,'s'):
  SESSION.s=requests.Session();SESSION.s.headers.update({'User-Agent':UA})
 return SESSION.s

def api(host,params,pace=.12):
 """Cached, paced MediaWiki/Wikidata request. Public read-only endpoints only."""
 params=dict(params,format='json',formatversion=2)
 key=hashlib.sha256((host+json.dumps(params,sort_keys=True)).encode()).hexdigest()[:24]
 path=CACHE/(key+'.json')
 if path.exists():
  try:return json.loads(path.read_text())
  except Exception:path.unlink()
 for attempt in range(4):
  with LOCK:
   wait=max(0.,THROTTLE['next']-time.monotonic());THROTTLE['next']=max(THROTTLE['next'],time.monotonic())+pace
  if wait:time.sleep(wait)
  try:
   r=session().get(f'https://{host}/w/api.php',params=params,timeout=(6,30))
   if r.status_code in (429,503):
    time.sleep(min(60.,float(r.headers.get('Retry-After',5))*(attempt+1)));continue
   r.raise_for_status();data=r.json()
   CACHE.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
   return data
  except Exception:
   if attempt==3:raise
   time.sleep(2*(attempt+1))

def claims(qids):
 """Batch Wikidata claims/labels for candidate entities."""
 out={}
 for i in range(0,len(qids),40):
  chunk=[q for q in qids[i:i+40] if q]
  if not chunk:continue
  data=api('www.wikidata.org',{'action':'wbgetentities','ids':'|'.join(chunk),'props':'claims|labels|aliases|sitelinks',
   'languages':'en','sitefilter':'enwiki|commonswiki'})
  out.update(data.get('entities',{}))
 return out

def values(entity,prop):
 out=[]
 for c in entity.get('claims',{}).get(prop,[]):
  snak=c.get('mainsnak',{})
  if snak.get('snaktype')!='value':continue
  v=snak['datavalue']['value']
  out.append(v['id'] if isinstance(v,dict) and 'id' in v else v)
 return out

def birth_year(entity):
 for v in entity.get('claims',{}).get('P569',[]):
  snak=v.get('mainsnak',{})
  if snak.get('snaktype')=='value':
   m=re.match(r'[+-](\d{4})',snak['datavalue']['value']['time'])
   if m:return int(m.group(1))
 return None

def club_labels(qids,cache):
 need=[q for q in qids if q not in cache]
 for i in range(0,len(need),40):
  data=api('www.wikidata.org',{'action':'wbgetentities','ids':'|'.join(need[i:i+40]),'props':'labels|aliases','languages':'en'})
  for q,e in data.get('entities',{}).items():
   names=[e.get('labels',{}).get('en',{}).get('value','')]+[a['value'] for a in e.get('aliases',{}).get('en',[])]
   cache[q]=[n for n in names if n]
 for q in qids:cache.setdefault(q,[])
 return cache

def club_match(team,labels):
 """Roster club strings are abbreviated ('Atl. Madrid', 'Man Utd'); match token prefixes."""
 want=[ABBREV.get(t,t) for t in tokens(team)]
 want=[t for t in want if t]
 if not want:return False
 for label in labels:
  have=[t for t in tokens(label) if t not in {'fc','cf','sv','ac','as','club','de','futbol','football','fussball','calcio'}]
  if all(any(h.startswith(t) or t.startswith(h) for h in have) for t in want):return True
 return False

def wikipedia_candidates(name):
 """Prefer the English Wikipedia article a roster spelling redirects to, then a name search."""
 qids=[]
 data=api('en.wikipedia.org',{'action':'query','titles':name,'redirects':1,'prop':'pageprops'})
 for page in data.get('query',{}).get('pages',[]):
  props=page.get('pageprops') or {}
  if page.get('missing') or 'disambiguation' in props:continue
  if props.get('wikibase_item'):qids.append(props['wikibase_item'])
 data=api('en.wikipedia.org',{'action':'query','generator':'search','gsrsearch':f'"{name}" footballer','gsrlimit':8,'prop':'pageprops'})
 for page in data.get('query',{}).get('pages',[]) if data.get('query') else []:
  props=page.get('pageprops') or {}
  if 'disambiguation' in props:continue
  if props.get('wikibase_item') and props['wikibase_item'] not in qids:qids.append(props['wikibase_item'])
 data=api('www.wikidata.org',{'action':'wbsearchentities','search':name,'language':'en','uselang':'en','type':'item','limit':10})
 for hit in data.get('search',[]):
  if hit['id'] not in qids:qids.append(hit['id'])
 return qids

def resolve(player,clubs):
 """Return (entity, decision) for one roster player; entity is None when ambiguous/unfound."""
 name=player['name'];qids=wikipedia_candidates(name)
 if not qids:return None,{'pid':player['pid'],'name':name,'reason':'No Wikidata candidate for roster spelling'}
 entities=claims(qids)
 footballers=[]
 for qid in qids:
  e=entities.get(qid)
  if not e or FOOTBALLER not in values(e,'P106'):continue
  year=birth_year(e)
  if year is not None and not 1975<=year<=2001:continue
  labels=[e.get('labels',{}).get('en',{}).get('value','')]+[a['value'] for a in e.get('aliases',{}).get('en',[])]
  exact=any(norm(l)==norm(name) for l in labels if l)
  footballers.append({'qid':qid,'entity':e,'labels':[l for l in labels if l],'exact_name':exact,'clubs':values(e,'P54')})
 if not footballers:return None,{'pid':player['pid'],'name':name,'reason':'No association-football-player entity among candidates'}
 club_labels([q for f in footballers for q in f['clubs']],clubs)
 for f in footballers:f['club_match']=club_match(player['team'],[l for q in f['clubs'] for l in clubs.get(q,[])])
 for rule,pool in [('exact name and club',[f for f in footballers if f['exact_name'] and f['club_match']]),
                   ('club history',[f for f in footballers if f['club_match']]),
                   ('unique exact name',[f for f in footballers if f['exact_name']]),
                   ('single candidate',footballers)]:
  if len(pool)==1:return pool[0],{'pid':player['pid'],'name':name,'qid':pool[0]['qid'],'rule':rule,
    'candidates':len(footballers),'club_match':pool[0]['club_match'],'exact_name':pool[0]['exact_name']}
 return None,{'pid':player['pid'],'name':name,'reason':'Ambiguous: several football entities match the roster name',
  'candidates':[f['qid'] for f in footballers]}

def photos(match,limit=48):
 """P18 claim, Commons person-category files and the enwiki lead image, with file provenance."""
 e=match['entity'];files=[]
 for f in values(e,'P18'):files.append(('File:'+f,'wikidata_image'))
 category=next(iter(values(e,'P373')),None)
 if not category:
  link=e.get('sitelinks',{}).get('commonswiki',{}).get('title','')
  category=link.split('Category:',1)[1] if link.startswith('Category:') else None
 if category:
  data=api('commons.wikimedia.org',{'action':'query','list':'categorymembers','cmtitle':'Category:'+category,
   'cmtype':'file','cmlimit':limit})
  for member in data.get('query',{}).get('categorymembers',[]):files.append((member['title'],'commons_person_category'))
 title=e.get('sitelinks',{}).get('enwiki',{}).get('title')
 if title:
  data=api('en.wikipedia.org',{'action':'query','titles':title,'prop':'pageimages','piprop':'name'})
  for page in data.get('query',{}).get('pages',[]):
   if page.get('pageimage'):files.append(('File:'+page['pageimage'],'wikipedia_lead_image'))
 seen={};order=[]
 for title,role in files:
  if title not in seen:seen[title]=role;order.append(title)
 info={}
 for i in range(0,len(order),40):
  data=api('commons.wikimedia.org',{'action':'query','titles':'|'.join(order[i:i+40]),'prop':'imageinfo',
   'iiprop':'url|size|mime|extmetadata','iiurlwidth':1024})
  for page in data.get('query',{}).get('pages',[]):
   if page.get('imageinfo'):info[page['title']]=page['imageinfo'][0]
 out=[]
 for title in order:
  ii=info.get(title)
  if not ii or ii.get('mime') not in {'image/jpeg','image/png','image/webp'}:continue
  url=ii.get('thumburl') or ii.get('url')
  out.append({'file':title,'image_role':seen[title],'image_url':url.split('?')[0],
   'original_url':ii['url'].split('?')[0],'original_width':ii.get('width'),'original_height':ii.get('height'),
   'descriptionurl':ii.get('descriptionurl'),'commons_category':category})
 return out
