"""Mine every saved page for every roster player, not just the one whose search found it.

`collect_pages` fetched a page only when its title or address named the player whose search
record listed it, and then looked for that one player's name in the image captions. But a
search record holds the merged results of a four-player batch, and a match report names a
dozen players in its captions, so most of what was already downloaded was never examined.
This re-reads the page cache against the whole roster at no network cost, and fetches the
pages the old rule skipped. Caption evidence is unchanged: an image is a candidate for a
player only when its own alt text names them.
"""
import hashlib,json,re,threading,time,unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import urljoin,urlparse
from bs4 import BeautifulSoup
import requests
from soccer_gif_reid.common import atomic_json,write_manifest
from .collect_pages import COOLDOWNS,HOST_LOCK,named,norm
from .metadata import DATA

BLOCKED=('reddit.','facebook.','instagram.','pinterest.','shutterstock.','gettyimages.','alamy.','tiktok.')
USER_AGENT='SoccerReIDResearch/0.1 (non-commercial player re-identification research)'


def tokens(s):
 return [t for t in re.split(r'[^a-z0-9]+',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode()) if t]


def surname_index(players):
 """Map a distinctive name token to the players who could own it, to avoid 1,741 tests per caption."""
 index=defaultdict(list)
 for p in players:
  parts=[t for t in tokens(p['name']) if len(t)>=4]
  for t in (parts[-1:] or tokens(p['name'])[-1:]):index[t].append(p)
 return index


def fetch(url):
 key=hashlib.sha256(url.encode()).hexdigest()[:20];cache=DATA/'page_cache'/(key+'.json')
 if cache.exists():
  try:return json.loads(cache.read_text()),None
  except Exception:cache.unlink()
 host=urlparse(url).netloc
 try:
  with HOST_LOCK:
   if COOLDOWNS.get(host,0)>time.time():raise ValueError('Host cooldown after HTTP 429')
  response=requests.get(url,timeout=(5,18),headers={'User-Agent':USER_AGENT})
  if response.status_code==429:
   try:delay=max(60.,float(response.headers.get('Retry-After','300')))
   except ValueError:delay=300.
   with HOST_LOCK:COOLDOWNS[host]=time.time()+delay
  response.raise_for_status()
  if 'html' not in response.headers.get('content-type',''):raise ValueError('Not an HTML page')
  soup=BeautifulSoup(response.text,'html.parser')
  title=soup.title.get_text(' ',strip=True) if soup.title else url
  images=[]
  for meta in soup.select('meta[property="og:image"],meta[name="twitter:image"],meta[property="og:image:url"]'):
   if meta.get('content'):images.append({'url':urljoin(url,meta['content']),'alt':title,'role':'page_featured_image'})
  for im in soup.find_all('img'):
   alt=im.get('alt','');src=im.get('data-src') or im.get('src')
   if src and alt:images.append({'url':urljoin(url,src),'alt':alt,'role':'captioned_image'})
  content={'title':title,'images':images,'page_sha256':hashlib.sha256(response.content).hexdigest(),'fetched_utc':time.time()}
  cache.parent.mkdir(exist_ok=True);atomic_json(cache,content)
  return content,None
 except Exception as e:return None,{'source_url':url,'reason':str(e).splitlines()[0]}


def mine(url,content,titles,index):
 """Every candidate this page offers: captions that name a player, plus its own hero image."""
 rows=[];address=urlparse(url).path
 for im in content['images']:
  flat=norm(im['alt']);people=[]
  if im['role']=='captioned_image':
   for token in set(tokens(im['alt'])):
    for p in index.get(token,[]):
     surname=tokens(p['name'])[-1]
     if named(p['name'],im['alt']) or (len(surname)>=5 and surname in flat):people.append(p)
  else:
   people=[p for p in titles if named(p['name'],address) or named(p['name'],im['url'])]
  image_url=im['url']
  if not image_url.startswith(('https://','http://')):continue
  for p in {x['pid']:x for x in people}.values():
   key=hashlib.sha256(image_url.encode()).hexdigest()[:16]
   rows.append({'pid':p['pid'],'name':p['name'],'team':p['team'],'candidate_id':p['pid']+'_'+key,
    'path':str(DATA/'web_candidates'/p['pid']/(key+'.img')),'image_url':image_url,
    'source_url':url,'source_title':content['title'],'search_description':im['alt'],'image_alt':im['alt'],
    'reference_type':'named_web_page','page_sha256':content['page_sha256'],'image_role':im['role']})
 return rows


def main(workers=6):
 players=json.loads((DATA/'selected_players.json').read_text())
 index=surname_index(players)
 urls=set()
 for file in sorted((DATA/'page_search').glob('*.json')):
  for source in json.loads(file.read_text())['pages']:
   url=source['source_url'];host=urlparse(url).netloc.lower()
   if any(t in host for t in BLOCKED) or url.lower().endswith('.pdf'):continue
   urls.add(url)
 existing={r['candidate_id']:r for r in map(json.loads,(DATA/'page_candidates.jsonl').read_text().splitlines())} \
   if (DATA/'page_candidates.jsonl').exists() else {}
 before=len(existing);errors=[];done=0
 with ThreadPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(fetch,url):url for url in sorted(urls)}
  for future in as_completed(futures):
   url=futures[future];content,error=future.result();done+=1
   if error:errors.append(error)
   else:
    for row in mine(url,content,players,index):existing.setdefault(row['candidate_id'],row)
   if done%250==0:
    write_manifest(DATA/'page_candidates.jsonl',list(existing.values()))
    print(f'{done}/{len(urls)} pages, candidates {len(existing)} (+{len(existing)-before}), errors {len(errors)}',flush=True)
 write_manifest(DATA/'page_candidates.jsonl',list(existing.values()))
 atomic_json(DATA/'page_mining_errors.json',errors)
 audit={'pages_considered':len(urls),'page_fetch_errors':len(errors),
  'candidates_before':before,'candidates_after':len(existing),'new_candidates':len(existing)-before,
  'players_covered':len({r['pid'] for r in existing.values()}),
  'rule':"An image is a candidate for a player when its own alt text names them, or when it is the page's hero image and the page address or image URL names them. Pages are no longer restricted to the player whose search record listed them."}
 atomic_json(DATA/'page_mining_audit.json',audit)
 print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
