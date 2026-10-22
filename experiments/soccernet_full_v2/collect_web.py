"""Download source-linked candidates from saved image-search results for review."""
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import threading
import time
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageOps
import requests

from soccer_gif_reid.common import atomic_json, digest_file, write_manifest
from .metadata import DATA, ROOT
from experiments.arg_fra_expanded_gallery_v1.collect import dhash
REPORT=ROOT/"reports/soccernet_full_v2"
BLOCKED_HOSTS = {}
USER_AGENT = 'SoccerReIDResearch/0.1 (non-commercial player re-identification research)'
# Hosts that reject parallel traffic and are collected by the paced single stream in
# warm_downloads instead. Fetching them here would turn one burst into permanent errors.
PACED_HOSTS = ('wikimedia.org', 'wikipedia.org')
HOST_LOCK = threading.Lock()


def inventory():
    people=json.loads((DATA/'selected_players.json').read_text())
    all_rows=[]
    page_rows=defaultdict(list)
    for extra in [DATA/'page_candidates.jsonl',DATA/'wiki_candidates.jsonl']:
        if extra.exists():
            for r in map(json.loads,extra.read_text().splitlines()):page_rows[r['pid']].append(r)
    for player in people:
        urls=set();rows=[]
        for file in sorted((DATA/'web_search').glob(player['pid']+'*.txt')):
            for block in re.split(r'\n?-{20,}\n?',file.read_text()):
                heading=re.search(r'^(.+) \((https?://[^\n]+)\)',block)
                image=re.search(r'Image URL:\s*(https?://[^\s#]+)',block)
                if not heading or not image:continue
                url=image.group(1)
                if url in urls:continue
                urls.add(url)
                key=hashlib.sha256(url.encode()).hexdigest()[:16]
                rows.append({'pid':player['pid'],'name':player['name'],'team':player['team'],
                    'candidate_id':player['pid']+'_'+key,'display_index':len(rows),
                    'path':str(DATA/'web_candidates'/player['pid']/(key+'.img')),
                    'image_url':url,'source_url':heading.group(2),'source_title':heading.group(1),
                    'search_record':str(file),'search_description':block[image.end():].lstrip('# ').strip(),
                    'reference_type':'fullbody_search' if '_fullbody' in file.name else 'portrait_search'})
        for r in page_rows[player['pid']]:
            if r['image_url'] not in urls:
                urls.add(r['image_url']);rows.append(dict(r,display_index=len(rows)))
        all_rows.extend(rows)
    write_manifest(DATA/'web_candidates.jsonl',all_rows)
    return all_rows


def fetch(row):
    row=dict(row);path=Path(row['path'])
    try:
        if not path.exists():
            old=ROOT/'data/soccernet_named_v1/web_candidates'/row['pid']/path.name
            if old.exists():
                path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,path)
        if not path.exists():
            host=urlparse(row['image_url']).netloc
            with HOST_LOCK:
                if BLOCKED_HOSTS.get(host,0)>time.time():raise ValueError('Host cooldown after HTTP 429')
            with requests.get(row['image_url'],headers={'User-Agent':USER_AGENT},timeout=(5,15),stream=True) as r:
                if r.status_code==429:
                    try:delay=max(60.,float(r.headers.get('Retry-After','300')))
                    except ValueError:delay=300.
                    with HOST_LOCK:BLOCKED_HOSTS[host]=time.time()+delay
                r.raise_for_status()
                data=bytearray()
                for part in r.iter_content(262144):
                    data.extend(part)
                    if len(data)>25000000:raise ValueError('Source exceeds 25 MB candidate limit')
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                if min(im.size)<180:raise ValueError('Source image below 180 pixels on short side')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        with Image.open(path) as im:
            original_format=im.format
            if original_format not in {'JPEG','PNG','WEBP','BMP'}:raise ValueError('Unsupported reference format '+str(original_format))
            if min(im.size)<180:raise ValueError('Source image below 180 pixels on short side')
            im=ImageOps.exif_transpose(im)
            row.update(width=im.width,height=im.height,sha256=digest_file(path),
                dhash=f'{dhash(im):016x}',format=original_format)
        return row,None
    except Exception as e:
        return None,{'candidate_id':row['candidate_id'],'image_url':row['image_url'],
                     'reason':str(e).splitlines()[0]}


def sheets(rows):
    out=REPORT/'candidate_sheets';out.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    for r in rows:grouped[r['pid']].append(r)
    for pid,items in grouped.items():
        items.sort(key=lambda r:r['display_index'])
        canvas=Image.new('RGB',(1200,325*((len(items)+5)//6)),'white');d=ImageDraw.Draw(canvas)
        for i,row in enumerate(items):
            x=i%6*200;y=i//6*325
            with Image.open(row['path']) as im:
                im=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(190,270))
                canvas.paste(im,(x+(200-im.width)//2,y+25))
            d.text((x+4,y+3),f'{row["display_index"]:02d} {row["width"]}x{row["height"]}',fill='black')
            d.text((x+4,y+300),urlparse(row['source_url']).netloc[:27],fill='black')
        canvas.save(out/(pid+'.jpg'),quality=94)


def main(cached_only=False,skip_paced_hosts=True):
    """Register every candidate whose bytes are on disk; optionally fetch nothing new.

    `cached_only` exists because the entity-linked candidates are fetched by the paced
    `warm_downloads` stream: registering them must not send eight parallel threads back at a
    host that is already rejecting bursts, and a burst of refusals would otherwise be recorded
    as permanent download errors.
    """
    rows=inventory();good=[];errors=json.loads((DATA/'web_download_errors.json').read_text()) if (DATA/'web_download_errors.json').exists() else []
    previous={r['candidate_id'] for r in json.loads((DATA/'web_download_errors.json').read_text())} if (DATA/'web_download_errors.json').exists() else set()
    rows=[r for r in rows if r['candidate_id'] not in previous or Path(r['path']).exists()]
    if cached_only:rows=[r for r in rows if Path(r['path']).exists()]
    if skip_paced_hosts:
        rows=[r for r in rows if Path(r['path']).exists()
              or not any(h in urlparse(r['image_url']).netloc for h in PACED_HOSTS)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fetch,row) for row in rows]
        for i,future in enumerate(as_completed(futures),1):
            row,error=future.result()
            if row:good.append(row)
            if error:errors.append(error)
            if i%40==0:
                print(f'{i}/{len(rows)} attempts: {len(good)} downloaded',flush=True)
                write_manifest(DATA/'web_downloaded.jsonl',sorted(good,key=lambda r:(r['pid'],r['display_index'])))
    good.sort(key=lambda r:(r['pid'],r['display_index']))
    write_manifest(DATA/'web_downloaded.jsonl',good);atomic_json(DATA/'web_download_errors.json',errors)
    # Final screened-reference sheets are produced by screen_references; avoid redrawing every raw candidate each incremental pass.
    print(json.dumps({'downloaded_images':len(good),'players':len({r['pid'] for r in good}),'errors':len(errors)}),flush=True)


if __name__=='__main__':
    import sys
    main(cached_only='--cached-only' in sys.argv,skip_paced_hosts='--include-paced-hosts' not in sys.argv)
