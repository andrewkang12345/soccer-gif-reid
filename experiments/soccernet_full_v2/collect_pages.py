"""Find reference candidates in named public pages, preserving image-level provenance.
No authentication, paywall bypass, or challenge solving. Failed/limited sources are deferred.
"""
import json,re,hashlib,unicodedata,time,threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import urljoin,urlparse
from bs4 import BeautifulSoup
import requests
from soccer_gif_reid.common import atomic_json,write_manifest
from .metadata import DATA,ROOT

COOLDOWNS={}
HOST_LOCK=threading.Lock()

def norm(s):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s.lower()).encode('ascii','ignore').decode())
def named(name,text):
 tokens=[norm(t) for t in name.split() if len(norm(t))>=3];text=norm(text)
 if name=='Juan Manuel Mata':tokens=['juan','mata']
 return bool(tokens) and all(t in text for t in tokens)
def page(job):
 p,source=job;url=source['source_url'];key=hashlib.sha256(url.encode()).hexdigest()[:20]
 cache=DATA/'page_cache'/(key+'.json');cache.parent.mkdir(exist_ok=True)
 try:
  if cache.exists():content=json.loads(cache.read_text())
  else:
   host=urlparse(url).netloc
   with HOST_LOCK:
    if COOLDOWNS.get(host,0)>time.time():raise ValueError('Host cooldown after HTTP 429')
   response=requests.get(url,timeout=(5,18),headers={'User-Agent':'Mozilla/5.0 (compatible; SoccerReIDResearch/0.1)'})
   if response.status_code==429:
    try:delay=max(60.,float(response.headers.get('Retry-After','300')))
    except ValueError:delay=300.
    with HOST_LOCK:COOLDOWNS[host]=time.time()+delay
   response.raise_for_status()
   if 'html' not in response.headers.get('content-type',''):raise ValueError('Not an HTML page')
   soup=BeautifulSoup(response.text,'html.parser');title=soup.title.get_text(' ',strip=True) if soup.title else source['source_title']
   images=[]
   for meta in soup.select('meta[property="og:image"],meta[name="twitter:image"],meta[property="og:image:url"]'):
    if meta.get('content'):images.append({'url':urljoin(url,meta['content']),'alt':title,'role':'page_featured_image'})
   for im in soup.find_all('img'):
    alt=im.get('alt','');src=im.get('data-src') or im.get('src')
    if src and alt:images.append({'url':urljoin(url,src),'alt':alt,'role':'captioned_image'})
   content={'title':title,'images':images,'page_sha256':hashlib.sha256(response.content).hexdigest(),'fetched_utc':time.time()}
   atomic_json(cache,content)
  results=[];seen=set()
  surname=norm(p['name'].split()[-1])
  for im in content['images']:
   if im['role']=='captioned_image' and not (named(p['name'],im['alt']) or (len(surname)>=5 and surname in norm(im['alt']))):continue
   image_url=im['url']
   if not image_url.startswith(('https://','http://')) or image_url in seen:continue
   seen.add(image_url);imagekey=hashlib.sha256(image_url.encode()).hexdigest()[:16]
   results.append({'pid':p['pid'],'name':p['name'],'team':p['team'],'candidate_id':p['pid']+'_'+imagekey,
    'path':str(DATA/'web_candidates'/p['pid']/(imagekey+'.img')),'image_url':image_url,
    **source,'search_description':im['alt'],'image_alt':im['alt'],'reference_type':'named_web_page',
    'page_sha256':content['page_sha256'],'image_role':im['role']})
  return results,None
 except Exception as e:return [],{'pid':p['pid'],'source_url':url,'reason':str(e).splitlines()[0]}
def main():
 jobs=[];seen=set()
 for file in sorted((DATA/'page_search').glob('*.json')):
  record=json.loads(file.read_text());p=record['player']
  for source in record['pages']:
   url=source['source_url'];host=urlparse(url).netloc.lower()
   if (p['pid'],url) in seen or not named(p['name'],source['source_title']+' '+url):continue
   if any(t in host for t in ['reddit.','facebook.','instagram.','pinterest.','shutterstock.','gettyimages.','alamy.','tiktok.']) or url.lower().endswith('.pdf'):continue
   seen.add((p['pid'],url));jobs.append((p,source))
 old=[json.loads(s) for s in (DATA/'page_candidates.jsonl').read_text().splitlines()] if (DATA/'page_candidates.jsonl').exists() else []
 completed=json.loads((DATA/'pages_processed.json').read_text()) if (DATA/'pages_processed.json').exists() else []
 done={tuple(x) for x in completed};jobs=[j for j in jobs if (j[0]['pid'],j[1]['source_url']) not in done]
 errors=[];rows={r['candidate_id']:r for r in old}
 with ThreadPoolExecutor(max_workers=6) as pool:
  futures={pool.submit(page,j):j for j in jobs}
  for i,f in enumerate(as_completed(futures),1):
   rs,err=f.result();j=futures[f];completed.append([j[0]['pid'],j[1]['source_url']])
   for r in rs:rows[r['candidate_id']]=r
   if err:errors.append(err)
   if i%40==0:
    write_manifest(DATA/'page_candidates.jsonl',list(rows.values()));atomic_json(DATA/'pages_processed.json',completed);print(i,len(jobs),'candidates',len(rows),'errors',len(errors),flush=True)
 write_manifest(DATA/'page_candidates.jsonl',list(rows.values()));atomic_json(DATA/'pages_processed.json',completed);atomic_json(DATA/'page_fetch_errors.json',errors)
 print('done pages',len(jobs),'total candidates',len(rows),flush=True)
if __name__=='__main__':main()
