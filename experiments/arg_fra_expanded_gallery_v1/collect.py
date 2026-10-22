"""Collect candidate web photographs from saved Commons metadata; review before use."""
import argparse
from collections import defaultdict
import hashlib
import html
import io
import json
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import requests

from soccer_gif_reid.common import atomic_json, digest_file, write_manifest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/arg_fra_expanded_gallery_v1'
REPORT = ROOT/'reports/arg_fra_expanded_gallery_v1'


def norm(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',text).lower() if not unicodedata.combining(c))


def text(value):
    return html.unescape(re.sub('<[^>]+>', '', value or ''))


def dhash(im):
    a=np.asarray(im.convert('L').resize((9,8)),dtype=np.int16)
    return sum(int(v)<<i for i,v in enumerate((a[:,1:]>a[:,:-1]).flat))


def candidates():
    people={}
    for folder in ['metadata','categories','supplemental']:
        for path in sorted((DATA/folder).glob('*.json')):
            j=json.loads(path.read_text());r=j['player'];pid=r['pid']
            people.setdefault(pid,{'player':r,'pages':{}})
            people[pid]['pages'].update(j.get('response',{}).get('query',{}).get('pages',{}))
    rows=[];exclusions=[]
    for pid,data in sorted(people.items()):
        player=data['player'];name=norm(player['name'])
        aliases=[name]
        if 'cristianromero' in pid:aliases=['cristian gabriel romero']
        if 'julianalvarez' in pid:aliases=['julian alvarez (footballer)']
        ranked=[]
        for page in data['pages'].values():
            if not page.get('imageinfo'):continue
            im=page['imageinfo'][0];em=im.get('extmetadata',{})
            title=page['title'];cats=norm(em.get('Categories',{}).get('value',''))
            desc=text(em.get('ImageDescription',{}).get('value',''))
            date=text(em.get('DateTimeOriginal',{}).get('value',''))
            reason=None
            if not any(a in cats for a in aliases):reason='No matching footballer category in source metadata'
            elif min(im.get('width',0),im.get('height',0))<250:reason='Source image below 250 pixels on short side'
            elif not re.search(r'\.(jpg|jpeg|png|webp)$',title,re.I):reason='Not a supported raster photograph'
            elif re.search(r'upscal|ai.generat|coloriz',cats,re.I):reason='Source identifies AI/upscaled/colourised derivative'
            elif date.startswith('2022-12-18') or '2022 fifa world cup final' in cats:reason='Same event as test clip'
            elif re.search(r'graffiti|mural|statue|signature|autograph|figurine|caricature|painting|drawing|wax figure|dressing room|academy sports|casa y barrio',norm(title+' '+cats)):reason='Not an independent player photograph'
            if reason:
                exclusions.append({'pid':pid,'title':title,'reason':reason});continue
            # Prefer named, vertically framed photographs and modest age difference.
            # This is metadata-based selection, never tested-model score selection.
            title_norm=norm(title)
            tokens=name.split();surname=' '.join(tokens[-2:]) if 'di maria' in name else tokens[-1]
            named=surname in title_norm
            ratio=im['height']/im['width']
            year_match=re.search(r'(19|20)\d{2}',date)
            year=int(year_match.group()) if year_match else 2022
            priority=(0 if named else 2)+(0 if ratio>=1.1 else 2)+(0 if 2016<=year<=2024 else 1)
            if 'cropped' in title_norm:priority-=.3
            ranked.append((priority,title,{'pid':pid,'name':player['name'],'team':player['team'],
                'title':title,'commons_page_id':page['pageid'],'image_url':im['url'],
                'download_url':im['url'],'source_url':im['descriptionurl'],
                'description':desc,'categories':em.get('Categories',{}).get('value',''),
                'date':date,'artist':text(em.get('Artist',{}).get('value','')),
                'license':em.get('LicenseShortName',{}).get('value',''),
                'license_url':em.get('LicenseUrl',{}).get('value',''),
                'source_width':im['width'],'source_height':im['height']}))
        # Original and differently cropped copies share a normalized source family.
        seen=set();selected=[]
        for _,title,row in sorted(ranked):
            family=re.sub(r'\([^)]*crop[^)]*\)','',norm(title))
            family=re.sub(r'\s+',' ',family).strip()
            if family in seen:continue
            seen.add(family);selected.append(row)
        for index,row in enumerate(selected):
            row['candidate_id']=f'{pid}_{row["commons_page_id"]}'
            row['display_index']=index
            row['path']=str(DATA/'candidates'/pid/f'{row["commons_page_id"]}{Path(row["title"]).suffix.lower()}')
            rows.append(row)
        print(player['name'],len(selected),'eligible candidates',flush=True)
    write_manifest(DATA/'candidates.jsonl',rows)
    atomic_json(DATA/'metadata_exclusions.json',exclusions)
    return rows


def download(rows,limit):
    counts=defaultdict(int);errors=[];successful=[]
    for row in rows:
        pid=row['pid']
        if counts[pid]>=limit:continue
        path=Path(row['path'])
        provenance=path.with_suffix(path.suffix+'.source.json')
        try:
            if not path.exists() or not provenance.exists():
                response=requests.get(row['download_url'],headers={'User-Agent':'SoccerReIDResearch/0.1'},timeout=30)
                if response.status_code==429:
                    print('Image rate limit; pausing 45 seconds',flush=True);time.sleep(45)
                    response=requests.get(row['download_url'],headers={'User-Agent':'SoccerReIDResearch/0.1'},timeout=30)
                response.raise_for_status()
                with Image.open(io.BytesIO(response.content)) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB')
                    if min(im.size)<200:raise ValueError('Downloaded image too small')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(response.content)
                atomic_json(provenance,{'download_url':row['download_url'],'source_url':row['source_url']})
                time.sleep(.5)
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im)
                row.update(width=im.width,height=im.height,dhash=f'{dhash(im):016x}',sha256=digest_file(path))
            successful.append(row);counts[pid]+=1
            if counts[pid] in (8,limit):
                print(pid,counts[pid],'downloaded',flush=True)
                write_manifest(DATA/'downloaded_candidates.jsonl',successful)
        except Exception as e:
            errors.append({'candidate_id':row['candidate_id'],'error':str(e).splitlines()[0]})
    write_manifest(DATA/'downloaded_candidates.jsonl',successful)
    atomic_json(DATA/'download_errors.json',errors)
    print('Downloaded',dict(counts),'errors',len(errors),flush=True)
    return successful


def sheets(rows):
    REPORT.mkdir(parents=True,exist_ok=True)
    people=defaultdict(list)
    for row in rows:people[row['pid']].append(row)
    for pid,items in people.items():
        canvas=Image.new('RGB',(1200,320*((len(items)+5)//6)), 'white');draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(items):
            x=(i%6)*200;y=(i//6)*320
            with Image.open(row['path']) as im:
                im=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(190,265));canvas.paste(im,(x+(200-im.width)//2,y+30))
            draw.text((x+5,y+5),f'{row["display_index"]:03d} '+row['date'][:10],fill='black')
            draw.text((x+5,y+300),row['title'][5:30],fill='black')
        canvas.save(REPORT/(pid+'.jpg'),quality=92)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--limit',type=int,default=14)
    p.add_argument('--inventory-only',action='store_true');a=p.parse_args()
    rows=candidates()
    if not a.inventory_only:sheets(download(rows,a.limit))


if __name__=='__main__':main()
