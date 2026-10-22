"""Re-cut validation and test game-first, so both evaluate the way the model is deployed.

The original split hashed players and games independently and kept a crop only when both
partitions agreed, which discarded about nine tenths of every evaluation player's crops:
validation ended up with 35 players and 139 crops, of which three players had the three
distinct references the freeze requires. This chooses validation games first, makes every
player appearing in them a validation player, and drops those players' crops from all other
games. Player, normalized-name and game disjointness are unchanged; the test split is left
exactly as it was, and the identities exposed by the earlier pilot and by Argentina-France
stay in training.
"""
import json,re,unicodedata
from collections import Counter,defaultdict
from datetime import datetime,timezone
from soccer_gif_reid.common import atomic_json,write_manifest
from .finalize import MIN_EVALUATION_REFERENCES,MIN_REFERENCES
from .metadata import DATA,ROOT


def norm(s):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(s).lower()).encode('ascii','ignore').decode())


def forced_train():
 """Identities already exposed by the pilot or by the Argentina-France development clip."""
 old=json.loads((ROOT/'data/soccernet_named_v1/players.json').read_text())
 pids={r['pid'] for r in old};names={norm(r['name']) for r in old}
 for r in map(json.loads,(ROOT/'data/arg_fra_expanded_gallery_v1/gallery.jsonl').read_text().splitlines()):
  names.add(norm(r['name']))
 return pids,names


def select(games,by_game,references,target,blocked,limit=None):
 """Greedily take the games that add the most reference-covered, unexposed players."""
 chosen=[];taken=set()
 while len(taken)<target and (limit is None or len(chosen)<limit):
  best=None
  for game in games:
   if game in chosen:continue
   gain={p for p in by_game[game] if p not in taken and p not in blocked and references.get(p,0)>=MIN_EVALUATION_REFERENCES}
   if best is None or len(gain)>len(best[1]) or (len(gain)==len(best[1]) and game<best[0]):best=(game,gain)
  if not best or not best[1]:break
  chosen.append(best[0]);taken|=best[1]
 return chosen,taken


def main(target=140,test_target=190,matches=12):
 # Always re-cut from the original hash partition so the step is idempotent and reversible.
 backup=DATA/'backup_prerepartition'
 source=backup if (backup/'masked_queries.jsonl').exists() else DATA
 rows=[json.loads(s) for s in (source/'masked_queries.jsonl').read_text().splitlines()]
 players={r['pid']:r for r in json.loads((source/'players.json').read_text())}
 references=Counter(json.loads(s)['pid'] for s in (DATA/'screened_gallery.jsonl').read_text().splitlines())
 pilot_pids,pilot_names=forced_train()
 by_game=defaultdict(set)
 for r in rows:by_game[r['game']].add(r['pid'])
 exposed={p for p in players if p in pilot_pids or norm(players[p]['name']) in pilot_names}
 blocked=set(exposed)
 candidates=sorted(by_game)
 # Test is chosen first and from the densest games, because a match-level evaluation needs many
 # players from one match: gallery is that match's roster, two kits, about twenty candidates.
 test_games,_=select(candidates,by_game,references,test_target,blocked,matches)
 test_games=set(test_games)
 test_players={p for g in test_games for p in by_game[g]}-blocked
 blocked=blocked|test_players|{p for p in players if norm(players[p]['name']) in {norm(players[x]['name']) for x in test_players}}
 candidates=[g for g in candidates if g not in test_games]
 val_games,_=select(candidates,by_game,references,target,blocked,matches)
 val_games=set(val_games)
 val_players={p for g in val_games for p in by_game[g]}-blocked
 # A validation name must not also identify a training player, so drop name collisions.
 name_counts=Counter(norm(players[p]['name']) for p in players)
 val_players={p for p in val_players if name_counts[norm(players[p]['name'])]==1}
 evaluation={'val':(val_games,val_players),'test':(test_games,test_players)}
 out=[];dropped=Counter()
 for r in rows:
  split=next((s for s,(g,p) in evaluation.items() if r['pid'] in p),None)
  game=next((s for s,(g,p) in evaluation.items() if r['game'] in g),None)
  if split and split==game:out.append(dict(r,split=split))
  elif split:dropped[f'{split} player outside its {split} games']+=1
  elif game:dropped[f'training player inside a {game} game']+=1
  else:out.append(dict(r,split='train'))
 splits=defaultdict(set)
 for r in out:splits[r['split']].add(r['pid'])
 for pid,p in players.items():
  p['split']='val' if pid in splits['val'] else 'test' if pid in splits['test'] else 'train'
 write_manifest(DATA/'masked_queries.jsonl',out)
 atomic_json(DATA/'players.json',list(players.values()))
 atomic_json(DATA/'selected_players.json',list(players.values()))
 games=json.loads((source/'games.json').read_text())
 for g in games:g['split']='val' if g['game'] in val_games else 'test' if g['game'] in test_games else 'train'
 atomic_json(DATA/'games.json',games)
 audit={'reassigned_utc':datetime.now(timezone.utc).isoformat(),'target_validation_players':target,'matches_per_evaluation_split':matches,
  'minimum_references':{'train':MIN_REFERENCES,'evaluation':MIN_EVALUATION_REFERENCES},
  'validation_games':sorted(val_games),'test_games':sorted(test_games),'dropped_crops':dict(dropped),
  'splits':{s:{'players':len(splits[s]),'crops':sum(r['split']==s for r in out),
   'games':len({r['game'] for r in out if r['split']==s}),
   'players_per_game':sorted((len({p for p in splits[s] if p in by_game[g]}) for g in ({'val':val_games,'test':test_games}.get(s,[]))),reverse=True),
   'players_with_enough_references':sum(1 for p in splits[s] if references.get(p,0)>=MIN_REFERENCES)} for s in ('train','val','test')},
  'rule':'Test games are chosen first and validation games from what remains, both before any player is assigned. Every unexposed player in an evaluation game becomes a player of that split and loses its crops in every other game, so each evaluation match carries most of its own roster and can be scored the way the model is deployed.',
  'protected_from_evaluation':len(exposed)}
 atomic_json(DATA/'repartition_audit.json',audit)
 print(json.dumps(audit['splits'],indent=2));print('dropped',dict(dropped),'val games',len(val_games))


if __name__=='__main__':main()
