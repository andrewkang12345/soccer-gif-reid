"""Fetch candidate reference bytes into the shared cache at a rate the public host allows.

Wikimedia's content host rejects bursts from a single address for a long time afterwards, so
this is a single patient stream that honours `Retry-After` exactly rather than retrying hard.
Candidates are ordered evaluation-split first and then round-robin across players by title
rank, so the number of players holding at least one usable reference grows as fast as the
budget allows instead of one player being completed at a time. It writes no manifest:
`collect_web` stays the only place that records what was downloaded.
"""
import io,json,sys,time
from collections import Counter
from pathlib import Path
import requests
from PIL import Image
from .body_preference import body_likelihood
from .metadata import DATA

USER_AGENT='SoccerReIDResearch/0.1 (non-commercial player re-identification research)'
SPLIT_ORDER={'val':0,'test':1,'train':2}


def deficits(minimum=3):
    """How many more screened references each player needs to become eligible."""
    path=DATA/'screened_gallery.jsonl'
    have=Counter(json.loads(line)['pid'] for line in path.read_text().splitlines()) if path.exists() else Counter()
    return {pid:max(0,minimum-have.get(pid,0)) for pid in
            {json.loads(line)['pid'] for line in (DATA/'masked_queries.jsonl').read_text().splitlines()}}


def ordered(rows,players):
    """Spend the budget where it converts into eligible players fastest.

    Eligibility needs three screened references, so a player sitting on two is one accepted photo
    from counting and a player sitting on none is at least three away. Ordering by that remaining
    deficit — smallest first, then round-robin within each group — turns a fixed number of slow
    downloads into the largest number of usable identities. Round-robin still matters inside a
    group so the budget is not spent completing one player at a time.
    """
    short=deficits()
    per={}
    for row in rows:per.setdefault(row['pid'],[]).append(row)
    # Within a player, fetch the candidates most likely to show a whole player first: they are the
    # kind the evaluation slightly prefers, and a throttled queue rarely reaches the tail.
    for rows_ in per.values():
        rows_.sort(key=lambda r:-body_likelihood(r.get('source_title'),r.get('search_description'),
                                                 r.get('image_url'),r.get('source_url')))
    out=[]
    for need in sorted({n for n in short.values() if n>0}):
        keyed=sorted((k,v) for k,v in per.items() if short.get(k)==need)
        for depth in range(max((len(v) for _,v in keyed),default=0)):
            out+=[items[depth] for _,items in keyed if depth<len(items)]
    listed={id(r) for r in out}
    return out+[r for r in rows if id(r) not in listed]


def store(path,data):
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if min(im.size)<180:return False
        if im.format not in {'JPEG','PNG','WEBP','BMP'}:return False
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return True


def main(name='wiki_candidates.jsonl',gap=6.0,budget_seconds=0,limit=0):
    gap,budget_seconds,limit=float(gap),float(budget_seconds),int(limit)
    players={r['pid']:r for r in json.loads((DATA/'selected_players.json').read_text())}
    rows=[json.loads(s) for s in (DATA/name).read_text().splitlines()]
    todo=[r for r in ordered(rows,players) if not Path(r['path']).exists()]
    if limit:todo=todo[:limit]
    session=requests.Session();session.headers.update({'User-Agent':USER_AGENT})
    began=time.monotonic();saved=throttled=rejected=failed=streak=0;index=0
    # The host escalates its penalty when a client keeps calling during a cooldown, so the
    # advertised Retry-After is waited out in full and the spacing adapts to what it allows.
    while index<len(todo):
        if budget_seconds and time.monotonic()-began>budget_seconds:break
        row=todo[index]
        try:
            response=session.get(row['image_url'],timeout=(6,90))
            if response.status_code==429:
                throttled+=1;streak=0;gap=min(15.,gap*1.2)
                try:wait=float(response.headers.get('Retry-After','30'))
                except ValueError:wait=30.
                print(f'throttled at {index}/{len(todo)}: waiting {wait:.0f}s, spacing now {gap:.1f}s',flush=True)
                time.sleep(wait+2.);continue
            response.raise_for_status()
            if len(response.content)>25000000:rejected+=1
            else:saved+=store(Path(row['path']),response.content) or 0
            streak+=1
        except Exception:failed+=1
        index+=1
        if streak>=20:gap=max(3.,gap*.85);streak=0
        time.sleep(gap)
        if index%50==0:
            elapsed=time.monotonic()-began
            print(f'{index}/{len(todo)} saved {saved} throttled {throttled} failed {failed} '
                  f'rejected {rejected} spacing {gap:.1f}s rate {60*saved/elapsed:.1f}/min',flush=True)
    print(json.dumps({'queued':len(todo),'saved':saved,'throttled':throttled,'failed':failed,
        'rejected':rejected,'seconds':round(time.monotonic()-began,1)}),flush=True)


if __name__=='__main__':main(*sys.argv[1:])
