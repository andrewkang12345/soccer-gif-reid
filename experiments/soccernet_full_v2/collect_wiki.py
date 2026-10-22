"""Link roster players to public Wikidata footballer entities and queue their free photos.

Identity evidence here is the entity link, not a page title: a roster name is accepted only
when exactly one association-football-player entity (born 1979-2000, with a free photo)
carries that normalized name, or when club history leaves exactly one candidate. Everything
else is dropped as ambiguous. Photos are the entity's own `image` claim plus the files in its
Commons person category; category files can still be group shots, so the existing
dominant-person and duplicate screens continue to apply afterwards.
"""
import hashlib,json,re,unicodedata
from collections import Counter,defaultdict
from urllib.parse import quote,unquote
from soccer_gif_reid.common import atomic_json,write_manifest
from .metadata import DATA
from .body_preference import body_likelihood
from .commons_harvest import listing

STOP={'fc','cf','sv','ac','as','ss','ssc','club','de','del','di','futbol','football','fussball','calcio','the','1','05','04','96'}
ABBREV={'utd':'united','atl':'atletico','ath':'athletic','dep':'deportivo','psg':'parissaintgermain',
 'gladbach':'monchengladbach','sp':'sporting','dyn':'dynamo','st':'saint'}


def norm(s):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode())
def tokens(s):return [t for t in re.split(r'[^a-z0-9]+',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode()) if t]


def club_match(team,labels):
 """Roster clubs are abbreviated ('Atl. Madrid', 'Man Utd'); compare token prefixes."""
 want=[ABBREV.get(t,t) for t in tokens(team) if t not in STOP]
 if not want:return False
 for label in labels:
  have=[t for t in tokens(label) if t not in STOP]
  if have and all(any(h.startswith(t) or t.startswith(h) for h in have) for t in want):return True
 return False


def index(entities):
 by_name=defaultdict(list)
 for e in entities:
  for form in [e['name']]+e.get('aliases',[]):
   key=norm(form)
   if key and e not in by_name[key]:by_name[key].append(e)
 return by_name


def link(player,by_name):
 candidates=by_name.get(norm(player['name']),[])
 if not candidates:return None,{'pid':player['pid'],'name':player['name'],'reason':'No Wikidata footballer entity with this name and a free photo'}
 if len(candidates)==1:return candidates[0],{'rule':'unique entity name','candidates':1}
 matched=[e for e in candidates if club_match(player['team'],e.get('clubs',[]))]
 if len(matched)==1:return matched[0],{'rule':'club history','candidates':len(candidates)}
 return None,{'pid':player['pid'],'name':player['name'],'reason':'Ambiguous: several football entities share this name',
  'candidates':[e['qid'] for e in candidates]}


GROUP_WORDS=('vs','versus','team','squad','training','celebrat','fans','trophy','stadium','lineup','line-up',
 'group','crowd','match','cup final','supporters','bench','handshake','wall','duel','tackle')


def title_score(name,title,role):
 """Rank Commons category files by how likely the title describes this player alone.

 Category order is upload order, which has no relation to usefulness, and the download
 budget is limited, so titles naming the player (and not several others) come first.
 Group shots that slip through are still removed by the dominant-person screen.
 """
 if role=='wikidata_image':return 1000
 low=' '+title.lower()+' ';parts=[t for t in tokens(name) if len(t)>=4]
 hits=sum(1 for t in parts if t in norm(title))
 others=max(0,sum(1 for word in re.findall(r'[A-Z][a-z]{3,}',title) if norm(word) not in {norm(t) for t in tokens(name)})-1)
 score=6*hits+3*(hits==len(parts) and bool(parts))
 # Measured on the frozen gallery: whole-player references retrieve at worst as well as
 # head-and-shoulders ones and possibly better, so they are preferred when choosing which of a
 # category's files to spend the download budget on. Portraits are still collected.
 score+=6*(body_likelihood(title)-.36)
 score-=3*sum(1 for w in GROUP_WORDS if w in low)
 return score-others


def filepath(title):
 """Direct Commons content URL for the canonical file (the upload host, not the wiki tier)."""
 name=title.split('File:',1)[-1].replace(' ','_');digest=hashlib.md5(name.encode()).hexdigest()
 return f'https://upload.wikimedia.org/wikipedia/commons/{digest[0]}/{digest[:2]}/'+quote(name)


def main(category_limit=40,per_player=10):
 players=json.loads((DATA/'selected_players.json').read_text())
 entities=json.loads((DATA/'wikidata_footballers.json').read_text())
 by_name=index(entities)
 linked={};rejected=[]
 for p in players:
  entity,decision=link(p,by_name)
  if entity:linked[p['pid']]=(p,entity,decision)
  else:rejected.append(decision)
 categories=[e['commons_category'] for _,e,_ in linked.values() if e.get('commons_category')]
 files=listing(categories,limit=category_limit) if categories else {}
 rows=[];seen=set()
 for pid,(p,entity,decision) in sorted(linked.items()):
  items=[(unquote(entity['image'].rsplit('/',1)[-1]),'wikidata_image')]
  for title in files.get(entity.get('commons_category') or '',[]):items.append((title,'commons_person_category'))
  order=sorted(range(len(items)),key=lambda i:(-title_score(p['name'],items[i][0],items[i][1]),i))
  for title,role in [items[i] for i in order[:per_player]]:
   name=title.split('File:',1)[-1].replace('_',' ')
   if not re.search(r'\.(jpg|jpeg|png|webp)$',name,re.I):continue
   url=filepath(name);key=hashlib.sha256(url.encode()).hexdigest()[:16]
   if (pid,key) in seen:continue
   seen.add((pid,key))
   rows.append({'pid':pid,'name':p['name'],'team':p['team'],'candidate_id':pid+'_'+key,
    'path':str(DATA/'web_candidates'/pid/(key+'.img')),'image_url':url,
    'source_url':'https://commons.wikimedia.org/wiki/'+quote(('File:'+name).replace(' ','_')),
    'source_title':'Wikimedia Commons '+name,'search_description':name,
    'reference_type':'wikidata_entity','image_role':role,'wikidata_qid':entity['qid'],
    'entity_label':entity['name'],'entity_match_rule':decision['rule'],
    'commons_category':entity.get('commons_category')})
 write_manifest(DATA/'wiki_candidates.jsonl',rows)
 audit={'roster_players':len(players),'linked_players':len(linked),'candidate_images':len(rows),
  'images_per_player':dict(Counter(Counter(r['pid'] for r in rows).values())),
  'match_rules':dict(Counter(d['rule'] for _,_,d in linked.values())),
  'rejection_reasons':dict(Counter(r['reason'] for r in rejected)),
  'entity_pool':len(entities),'commons_categories_listed':len(files),'category_file_limit':category_limit,
  'downloaded_per_player_limit':per_player,
  'ranking':'Commons category files are ordered by a title heuristic favouring the player\'s own name and penalising group/scene wording, then truncated to the per-player limit.',
  'image_note':'Reference bytes are the original Commons file as uploaded, fetched from the content host; no local crop, resize or re-encode.',
  'evidence':'Unique association-football-player entity by normalized name, or unique club-history match; entity link recorded per image.'}
 atomic_json(DATA/'wiki_link_audit.json',audit)
 atomic_json(DATA/'wiki_link_rejections.json',rejected)
 print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
