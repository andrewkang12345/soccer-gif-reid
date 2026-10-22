"""Mine club squad and player-index pages for the roster players still short of references.

Most players are not short because no photograph of them exists but because our search records
never pointed at one. A club's own squad page lists twenty to thirty players with a captioned
headshot each, on a host that already passes the football-context screen and has never throttled
us, so one fetch can serve a whole team. The club's own domain is recovered from candidates we
already collected rather than guessed, and the usual squad paths are tried in the languages these
sites actually use. Evidence is unchanged: an image counts for a player only when its own alt
text names them.

Current squad pages list a current squad, and these matches are from 2014-2017, so the yield is
limited to players still at the club and to archive pages that happen to be linked. Pages that
name nobody on the roster cost a request and are discarded.
"""
import json,re,unicodedata
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import urlparse
from soccer_gif_reid.common import atomic_json,write_manifest
from .metadata import DATA
from .mine_pages import fetch,mine,surname_index

PATHS=('','/team','/teams','/squad','/squads','/first-team','/firstteam','/players','/player',
 '/equipo','/plantilla','/jugadores','/plantel','/kader','/mannschaft','/spieler','/profis',
 '/rosa','/squadra','/giocatori','/effectif','/joueurs','/elenco','/spelers','/selectie',
 '/en/team','/en/squad','/en/players','/es/plantilla','/de/kader','/it/rosa','/fr/effectif')


def norm(s):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode())
def words(s):
 plain=re.sub('[^a-z0-9 ]',' ',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode())
 return [t for t in plain.split() if len(t)>=4]


def club_hosts():
 """Recover each club's own domain from candidates already collected, not from a guess."""
 counts=defaultdict(Counter)
 for line in (DATA/'page_candidates.jsonl').read_text().splitlines():
  r=json.loads(line);counts[r['team']][urlparse(r['source_url']).netloc.lower()]+=1
 chosen={}
 for team,hosts in counts.items():
  club=norm(team);tokens=words(team)
  for host,_ in hosts.most_common():
   labels=[l for l in host.split('.') if l not in ('www','com','net','org','co','uk')]
   label=labels[0] if labels else ''
   # The club name must open the domain, so "villa" cannot claim diariodesevilla.es.
   if (len(club)>=5 and club in norm(host)) or any(label.startswith(t) for t in tokens):
    chosen[team]=host;break
 return chosen


def main(workers=6,minimum=3):
 players=json.loads((DATA/'selected_players.json').read_text())
 have=Counter(json.loads(l)['pid'] for l in (DATA/'screened_gallery.jsonl').read_text().splitlines())
 short={p['pid'] for p in players if have.get(p['pid'],0)<minimum}
 hosts=club_hosts()
 index=surname_index(players)
 targets=sorted({f'https://{host}{path}' for team,host in hosts.items() for path in PATHS})
 existing={r['candidate_id']:r for r in map(json.loads,(DATA/'page_candidates.jsonl').read_text().splitlines())}
 before=len(existing);pages=0;errors=[];gained=set()
 with ThreadPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(fetch,url):url for url in targets}
  for future in as_completed(futures):
   url=futures[future];content,error=future.result();pages+=1
   if error:errors.append(error);continue
   for row in mine(url,content,players,index):
    if row['candidate_id'] not in existing:
     existing[row['candidate_id']]=row
     if row['pid'] in short:gained.add(row['pid'])
   if pages%200==0:print(f'{pages}/{len(targets)} pages, new candidates {len(existing)-before}, '
                        f'short players reached {len(gained)}',flush=True)
 write_manifest(DATA/'page_candidates.jsonl',list(existing.values()))
 audit={'clubs':len(hosts),'clubs_without_an_identifiable_host':len({p['team'] for p in players})-len(hosts),
  'pages_tried':len(targets),'page_errors':len(errors),'new_candidates':len(existing)-before,
  'players_short_of_references':len(short),'short_players_given_a_new_candidate':len(gained),
  'note':'Squad pages list a current squad while these matches are from 2014-2017, so a player who has since left is only reached through archive pages.'}
 atomic_json(DATA/'squad_mining_audit.json',audit)
 print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
