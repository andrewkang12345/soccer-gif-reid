"""Mine free match photo galleries for whole-body references of the twenty-one demo players.

Wikimedia's content host allows about six images before asking for a ten minute cooldown, which
makes a Commons rebuild an overnight job, and the public image proxy that would have sidestepped
it is refused by the same host. Ordinary news galleries are not throttled at all, and a match
gallery is the right shape of source for this problem: every photograph is a whole player in
motion at broadcast distance, and each caption names who is in it, so one page supplies
candidates for many players at once.

The 2022 final is excluded by hand. It is the clip the demonstration runs on, and a reference
photograph taken minutes from the query frame would measure memory of one afternoon rather than
recognition of a player.
"""
import hashlib,io,json,re,sys,unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin,urlparse
import requests
from bs4 import BeautifulSoup
from PIL import Image
from soccer_gif_reid.common import atomic_json,write_manifest
from .metadata import ROOT

DATA=ROOT/'data/arg_fra_expanded_gallery_v1'
CANDIDATES=DATA/'web_candidates.jsonl'
USER_AGENT='Mozilla/5.0 (compatible; SoccerReIDResearch/0.1)'
EXCLUDED='2022/12/18'  # the final itself: same event as the demonstration clip
PAGES=[
 'https://www.aljazeera.com/gallery/2026/7/16/photos-messis-argentina-stun-england-to-reach-world-cup-final-vs-spain',
 'https://www.aljazeera.com/gallery/2026/7/20/photos-spain-beat-argentina-to-win-second-world-cup-title',
 'https://www.aljazeera.com/gallery/2022/12/13/photos-argentina-batter-croatia-shoot-world-cup-glory',
 'https://www.aljazeera.com/gallery/2022/11/26/photos-argentina-vs-mexico',
 'https://www.aljazeera.com/gallery/2022/12/3/photos-argentina-vs-australia',
 'https://www.aljazeera.com/gallery/2022/12/9/photos-messis-argentina-defeat-netherlands-in-shootout',
 'https://www.aljazeera.com/gallery/2026/7/15/photos-france-fireworks-fade-as-spain-march-into-world-cup-final',
 'https://www.aljazeera.com/gallery/2022/12/14/photos-france-charge-into-final-to-end-moroccos-world-cup-dream',
 'https://www.aljazeera.com/gallery/2022/11/26/photos-france-vs-denmark',
 'https://www.aljazeera.com/gallery/2022/12/4/photos-giroud-mbappe-france-beat-poland-world-cup-2022',
 'https://www.aljazeera.com/gallery/2022/11/22/photos-giroud-equals-record-as-holders-france-beat-australia-4-1',
 'https://www.aljazeera.com/gallery/2022/12/16/qatar-2022-best-on-field-highlights-from-the-world-cup',
 'https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/world-cup-finals-best-images-photographs',
 'https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/best-photos-matchday-32',
 'https://www.fifa.com/en/tournaments/mens/worldcup/articles/players-celebrate-trophy-photos',
 'https://www.fifa.com/en/tournaments/mens/worldcup/articles/lionel-messi-argentina-photos',
 'https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/stars-portrait-photographs-messi-ronaldo-mbappe',
 'https://www.fifa.com/photos/galleries/raphael-varane-france-real-madrid-1998903',
 'https://www.dailyrecordnews.com/sports/world_cup/',
 'https://www.dailyrecordnews.com/sports/world_cup/argentina-wcup-soccer/image_b5c02efb-558c-5a19-b16e-7a1d1a5eb90e.html',
 'https://www.dailyrecordnews.com/ap_news/sports/argentina-wcup-soccer/image_af85bdd7-2271-5319-91f9-ba46d8cc4760.html',
 'https://www.dailyrecordnews.com/ap_news/sports/france-wcup-soccer/image_2a8861fb-33af-5074-8127-9a3b2c856fae.html',
 'https://www.dailyrecordnews.com/sports/world_cup/france-wcup-soccer/image_ee328086-2845-525d-bbfd-dd43bdc8545a.html',
 'https://www.dailyrecordnews.com/sports/world_cup/spain-france-wcup-soccer/image_5162761c-0bb2-5789-a94a-65f84e3f824b.html',
 'https://www.dailyrecordnews.com/sports/world_cup/aptopix-argentina-wcup-soccer/image_0d620f47-65ee-58cf-babc-747e980de880.html',
 'https://www.dailyrecordnews.com/ap_news/sports/argentina-wcup-soccer/image_b2cbe12d-b00d-5ad3-b2a1-b2b9e42e6c19.html',
 'https://www.bdtonline.com/sports/national_sports/argentina-wcup-soccer/image_c1bcb610-87a8-594d-b67d-afbe03060eee.html',
 'https://www.bdtonline.com/sports/national_sports/spain-france-wcup-soccer/image_c9482680-fa2a-53bf-8464-746661a992bb.html',
 'https://www.bdtonline.com/sports/national_sports/aptopix-argentina-wcup-soccer/image_8ceb8626-bfb4-5eb5-bdd6-8bf98bd989b9.html',
 'https://bdtonline.com/sports/national_sports/argentina-england-wcup-soccer/image_5eef5f35-7e10-578f-872d-a62c6e926124.html',
 'https://www.dailygazette.com/ap/sports/one-extraordinary-photo-frances-kylian-mbappe-the-center-of-attention-of-again/article_7459d6be-6551-5cb6-9977-c2992343b4a8.html',
 'https://www.purepeople.com/media/aurelien-tchouameni-au-match-de-football_m7306563',
]
# Captions use the name a commentator would use, which is not always the roster spelling.
ALIASES={'Emiliano Martínez':['emi martinez','emiliano martinez','dibu martinez'],
 'Ángel Di María':['di maria'],'Alexis Mac Allister':['mac allister'],
 'Nicolás Otamendi':['otamendi'],'Nahuel Molina':['nahuel molina'],
 'Cristian Romero':['cristian romero','cuti romero'],'Nicolás Tagliafico':['tagliafico'],
 'Rodrigo De Paul':['de paul'],'Julián Álvarez':['julian alvarez'],
 'Enzo Fernández':['enzo fernandez'],'Lionel Messi':['lionel messi','leo messi'],
 'Aurélien Tchouaméni':['tchouameni'],'Dayot Upamecano':['upamecano'],
 'Théo Hernandez':['theo hernandez'],'Jules Koundé':['kounde'],
 'Ousmane Dembélé':['dembele'],'Raphaël Varane':['varane'],'Adrien Rabiot':['rabiot'],
 'Kylian Mbappé':['mbappe'],'Antoine Griezmann':['griezmann'],'Olivier Giroud':['giroud']}


def flat(s):
 return re.sub(r'[^a-z0-9 ]',' ',unicodedata.normalize('NFKD',str(s).lower())
               .encode('ascii','ignore').decode())


def roster():
 rows=[json.loads(s) for s in (DATA/'gallery.jsonl').read_text().splitlines()]
 seen={}
 for r in rows:seen.setdefault(r['pid'],{'pid':r['pid'],'name':r['name'],'team':r['team']})
 people=list(seen.values())
 for p in people:
  p['aliases']=[flat(a) for a in ALIASES.get(p['name'],[flat(p['name'])])]
 return people


def owner(caption,people):
 """The single player a caption names, or nothing when it names none or several."""
 text=' '+flat(caption)+' '
 hits=[p for p in people if any(' '+a+' ' in text for a in p['aliases'])]
 return hits[0] if len(hits)==1 else None


def page(session,url):
 key=hashlib.sha256(url.encode()).hexdigest()[:20]
 cache=DATA/'web_page_cache'/(key+'.json');cache.parent.mkdir(parents=True,exist_ok=True)
 if cache.exists():return json.loads(cache.read_text())
 response=session.get(url,timeout=(6,25));response.raise_for_status()
 soup=BeautifulSoup(response.text,'html.parser')
 images=[]
 for im in soup.find_all('img'):
  src=im.get('data-src') or im.get('src');alt=im.get('alt','')
  if src and alt:images.append({'url':urljoin(url,src),'alt':alt})
 for figure in soup.find_all('figure'):
  im=figure.find('img');cap=figure.find('figcaption')
  if im and cap:
   src=im.get('data-src') or im.get('src')
   if src:images.append({'url':urljoin(url,src),'alt':cap.get_text(' ',strip=True)})
 content={'url':url,'title':soup.title.get_text(' ',strip=True) if soup.title else url,'images':images}
 cache.write_text(json.dumps(content,ensure_ascii=False));return content


def full_size(url):
 """Ask the site's own image service for the original rather than a list thumbnail."""
 return re.sub(r'([?&])(w|width|resize|quality)=[^&]*','\\1',url).rstrip('?&')


def siblings(session,url,limit=40):
 """Wire-photo pages link to their siblings, so one hop multiplies a single caption into a set."""
 try:response=session.get(url,timeout=(6,25));response.raise_for_status()
 except Exception:return []
 host=urlparse(url).netloc;out=[]
 for a in BeautifulSoup(response.text,'html.parser').find_all('a',href=True):
  link=urljoin(url,a['href'])
  if urlparse(link).netloc==host and re.search(r'/image_[0-9a-f-]+\.html$',link) and link not in out:
   out.append(link)
 return out[:limit]


def collect(workers=8,crawl=True):
 people=roster();session=requests.Session();session.headers.update({'User-Agent':USER_AGENT})
 found=defaultdict(list);pages=0;skipped=[]
 queue=list(PAGES)
 if crawl:
  for url in PAGES:
   if '/image_' in url or url.rstrip('/').endswith('world_cup'):
    queue+=[u for u in siblings(session,url) if u not in queue]
 for url in queue:
  if EXCLUDED in url:skipped.append(url);continue
  try:content=page(session,url)
  except Exception as e:print('page failed',url,str(e)[:70],flush=True);continue
  pages+=1
  for image in content['images']:
   who=owner(image['alt'],people)
   if who is None:continue
   found[who['pid']].append({'pid':who['pid'],'name':who['name'],'team':who['team'],
    'title':image['alt'],'image_url':full_size(image['url']),'download_url':full_size(image['url']),
    'source_url':content['url'],'description':image['alt'],'categories':'',
    'license':'News gallery image; research use only, not redistributed'})
  print(f'{urlparse(url).path.split("/")[-1][:52]:54s} running total '
        f'{sum(len(v) for v in found.values())}',flush=True)
 rows=[]
 for pid,items in sorted(found.items()):
  seen=set()
  for i,r in enumerate(items):
   if r['image_url'] in seen:continue
   seen.add(r['image_url'])
   r['candidate_id']=pid+'_web_'+hashlib.sha256(r['image_url'].encode()).hexdigest()[:10]
   r['display_index']=len(rows);r['body_score']=None
   r['path']=str(DATA/'web_candidates'/pid/(r['candidate_id'].split('_')[-1]+'.jpg'))
   rows.append(r)
 def one(row):
  path=Path(row['path'])
  if path.exists():return 'saved'
  try:
   response=session.get(row['download_url'],timeout=(6,40))
   response.raise_for_status()
   with Image.open(io.BytesIO(response.content)) as im:
    im.load()
    if min(im.size)<200:return 'rejected'
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content);return 'saved'
  except Exception:return 'failed'
 counts=defaultdict(int)
 with ThreadPoolExecutor(workers) as pool:
  for outcome in pool.map(one,rows):counts[outcome]+=1
 kept=[r for r in rows if Path(r['path']).exists()]
 write_manifest(CANDIDATES,kept)
 per={p['pid']:sum(1 for r in kept if r['pid']==p['pid']) for p in people}
 atomic_json(DATA/'web_candidates_audit.json',{'pages_mined':pages,'pages_skipped':skipped,
  'pages_queued':len(queue),
  'excluded_event':'2022 World Cup final, the same match as the demonstration clip',
  'candidates':len(kept),'downloads':dict(counts),'per_identity':per,
  'caption_rule':'A photograph counts for a player only when its own caption names that player and no other player on the roster.'})
 print(json.dumps({'pages':pages,'candidates':len(kept),'downloads':dict(counts),
  'without_any':[k for k,v in per.items() if v==0],'per_identity':per},indent=2),flush=True)


if __name__=='__main__':collect(*(int(a) for a in sys.argv[1:]))
