"""Prepare real soccer crop-to-web-photo inputs, without model-based selection."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import unicodedata

from soccer_gif_reid.common import atomic_json, digest_file, read_rgb, write_manifest

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/mnt/data/sports_reID')
OUT=ROOT/'data/arg_fra_roster_probe_v1'
REPORT=ROOT/'reports/released_gif_soccer_check'


def normalized(value):
    value=unicodedata.normalize('NFKD',value)
    return re.sub('[^a-z0-9]','',value.encode('ascii','ignore').decode().lower())


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'queries').mkdir(exist_ok=True)
    source=SOURCE/'groundTruth_AllTracking_ARG_FRA_183303'
    annotation=source/'_annotations.coco.json'
    coco=json.loads(annotation.read_text())
    refs=[];reference_map={}
    for path in sorted((SOURCE/'models/player_faces').glob('*.jpg')):
        team,number,name=path.stem.split('_',2)
        key=f'{int(number)}:{normalized(name)}'
        if key in reference_map:raise ValueError('Ambiguous reference naming')
        pid=f'{team}_{int(number):02d}_{normalized(name)}'
        im=read_rgb(path)
        r=dict(path=str(path),pid=pid,team=team,number=int(number),name=name.replace('_',' '),
               width=im.width,height=im.height,sha256=digest_file(path),domain='reference',split='test',
               label_source='existing sports_reID reference filename; semantic identity not independently verified')
        refs.append(r);reference_map[key]=r
    categories={c['id']:c['name'] for c in coco['categories']}
    images={im['id']:im for im in coco['images']}
    groups=defaultdict(list);excluded=[]
    for a in coco['annotations']:
        label=categories[a['category_id']]
        if label in ('referee','Player-Detection'):
            excluded.append(dict(annotation_id=a['id'],reason='not_named_player'));continue
        name=label.removeprefix('for tracker ').removeprefix('Goalkeeper ')
        number,person=name.split(' ',1)
        key=f'{int(number)}:{normalized(person)}'
        if key not in reference_map:raise ValueError(f'No roster reference for {label}')
        groups[a['image_id'],reference_map[key]['pid']].append((a,reference_map[key]))
    queries=[];ambiguous=[];image_cache={}
    for (image_id,pid),items in sorted(groups.items()):
        if len(items)>1:
            ambiguous.append(dict(image=images[image_id]['file_name'],pid=pid,
                annotations=[dict(id=a['id'],bbox=a['bbox'],category=categories[a['category_id']]) for a,r in items]))
            excluded.extend(dict(annotation_id=a['id'],reason='two_distinct_boxes_same_named_player_in_frame') for a,r in items)
            continue
        a,ref=items[0];meta=images[image_id]
        if image_id not in image_cache:image_cache[image_id]=read_rgb(source/meta['file_name'])
        im=image_cache[image_id];x,y,w,h=map(float,a['bbox'])
        if not all(math.isfinite(v) for v in [x,y,w,h]) or min(w,h)<=0:raise ValueError('Invalid bbox')
        box=[max(0,math.floor(x)),max(0,math.floor(y)),min(im.width,math.ceil(x+w)),min(im.height,math.ceil(y+h))]
        crop=im.crop(box)
        if min(crop.size)<2:raise ValueError('Empty clipped box')
        path=OUT/'queries'/f'annotation_{a["id"]:04d}.png';crop.save(path)
        match=re.search(r'-(\d+)_jpg',meta['file_name'])
        if not match:raise ValueError('Unknown source ordering')
        queries.append(dict(path=str(path),pid=pid,split='test',domain='query',
            annotation_id=a['id'],source_image=str(source/meta['file_name']),source_index=int(match[1]),
            bbox_original=a['bbox'],bbox_clamped=box,width=crop.width,height=crop.height,
            sha256=digest_file(path),team=ref['team'],name=ref['name'],label=categories[a['category_id']]))
    if {r['sha256'] for r in queries}&{r['sha256'] for r in refs}:raise ValueError('Query/gallery leakage')
    if set(r['pid'] for r in queries)-set(r['pid'] for r in refs):raise ValueError('Missing gallery positive')
    write_manifest(OUT/'manifest.jsonl',queries+refs)
    atomic_json(OUT/'annotation_conflicts.json',ambiguous)
    atomic_json(OUT/'excluded_annotations.json',excluded)
    report=dict(sport='soccer',source='ARG_FRA_183303',source_images=len(images),
        total_annotations=len(coco['annotations']),named_player_boxes=sum(len(v) for v in groups.values()),
        eligible_query_boxes=len(queries),query_identities=len(set(r['pid'] for r in queries)),
        clean_web_references=len(refs),gallery_identities=len(refs),
        exclusions=dict(Counter(r['reason'] for r in excluded)),
        annotations_sha256=digest_file(annotation),manifest_sha256=digest_file(OUT/'manifest.jsonl'),
        source_pixel_quality_retained=True,query_augmentation=False,
        query_protocol='Complete GT bounding boxes, floor/ceil clamped, PNG storage; no OCR, legibility or model-score filtering.',
        gallery_protocol='Existing web reference photos; filename labels inherited; both teams in retrieval gallery.',
        evaluation_scope='Previously inspected 31-frame soccer diagnostic, not a fresh test benchmark or end-to-end tracker evaluation.',
        label_limit='Five frame/identity conflicts excluded without choosing a box. Other identity-label errors may remain.')
    atomic_json(REPORT/'soccer_data_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
