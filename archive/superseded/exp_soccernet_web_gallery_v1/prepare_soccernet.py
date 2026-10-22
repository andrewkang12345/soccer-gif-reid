"""Prepare original-quality SoccerNet queries, preserving anonymous identity labels."""
from collections import defaultdict, Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from soccer_gif_reid.common import atomic_json, digest_file, write_manifest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/mnt/data/soccer_temporal_ocr/experiments/calibration_v4')


def main():
    out = ROOT/'data/soccernet_web_gallery_v1'
    out.mkdir(parents=True, exist_ok=True)
    report = ROOT/'reports/soccernet_web_gallery_v1'
    metadata = json.loads((SOURCE/'reports/test_metadata.json').read_text())
    by_game = defaultdict(list)
    for row in metadata:
        by_game[row['game_id']].append(row)
    chosen = []
    for game, rows in sorted(by_game.items()):
        rows.sort(key=lambda r: hashlib.sha256(('web-reid-v1:'+r['name']).encode()).hexdigest())
        chosen.extend(rows[:3])
    protocol = {'source':str(SOURCE/'data/gsr/test.zip'),
        'source_revision':'ce84ee7d9acb2f9fe999ceed2b38e303ffa5efb9',
        'source_sha256':'ac40080fadaebf3474477889576662de7970cae0cb375d7cd22d96f49aeefbfd',
        'selected_clips':chosen, 'split':'official test',
        'selection':'First three clips per anonymous match ID under SHA256(web-reid-v1:clip_name), before model scores',
        'queries':'Up to eight non-overlapping four-frame samples per labeled GT track, uniform over eligible candidates; stride four source frames',
        'gallery_target_per_player':8,
        'quality':'Original full GT bbox, floor/ceil clamped, lossless PNG. No legibility gate, OCR filtering, synthetic degradation or model-score selection.',
        'identity_status':'Anonymous clip/team/jersey keys only; verified name mapping required before web-image retrieval evaluation',
        'models':['runs/ap3d_no_quality_v2/clip/best.pt','runs/ap3d_no_quality_v2/ap3d/best.pt']}
    # Freeze protocol before reading pixels or running feature extraction.
    protocol_path = out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Frozen protocol differs')
    atomic_json(protocol_path, protocol)
    clips, frame_rows, mapping, exclusions = [], [], {}, Counter()
    with zipfile.ZipFile(SOURCE/'data/gsr/test.zip') as archive:
        for selected in chosen:
            data = json.loads(archive.read(selected['archive_member']))
            images = {str(i['image_id']):i for i in data['images']}
            tracks, duplicates = defaultdict(list), defaultdict(list)
            for ann in data['annotations']:
                if ann.get('supercategory') != 'object':continue
                attrs = ann['attributes']; number = str(attrs.get('jersey'))
                if attrs.get('role') not in ('player','goalkeeper'):
                    exclusions['non_player_annotations'] += 1; continue
                if not number.isdecimal() or attrs.get('team') not in ('left','right'):
                    exclusions['unknown_number_or_team_annotations'] += 1; continue
                key = (str(ann['image_id']), attrs['team'], str(int(number)))
                duplicates[key].append(ann)
                tracks[str(ann['track_id'])].append(ann)
            conflicting = {str(a['id']) for v in duplicates.values() if len(v)>1 for a in v}
            exclusions['contradictory_same_frame_identity_annotations'] += len(conflicting)
            requests = defaultdict(list)
            for track, anns in sorted(tracks.items()):
                identities = {(a['attributes']['team'],str(int(a['attributes']['jersey']))) for a in anns}
                if len(identities)!=1:
                    exclusions['tracks_with_conflicting_identity_attributes'] += 1; continue
                team, number = next(iter(identities))
                pid = f"{selected['name']}:{team}:{number}"
                mapping[pid] = {'anonymous_pid':pid,'game_id':selected['game_id'],
                    'clip':selected['name'],'half':selected['game_time_start'].split(' - ')[0],
                    'team_side':team,'jersey':number,'player_name':'','club_name':'','identity_source_url':''}
                indexed = {int(Path(images[str(a['image_id'])]['file_name']).stem):a for a in anns if str(a['id']) not in conflicting}
                candidates=[]; used=set()
                for start in sorted(indexed):
                    seq=[start+i*4 for i in range(4)]
                    if not all(i in indexed and i not in used for i in seq):continue
                    good=True
                    for f in seq:
                        a=indexed[f];im=images[str(a['image_id'])];b=a['bbox_image']
                        box=[max(0,math.floor(b['x'])),max(0,math.floor(b['y'])),
                             min(im['width'],math.ceil(b['x']+b['w'])),min(im['height'],math.ceil(b['y']+b['h']))]
                        if box[2]-box[0]<2 or box[3]-box[1]<2:good=False;break
                    if good:candidates.append(seq);used.update(seq)
                if not candidates:
                    exclusions['tracks_without_four_valid_observations'] += 1;continue
                picks=np.unique(np.linspace(0,len(candidates)-1,min(8,len(candidates)),dtype=int))
                for pick in picks:
                    seq=candidates[int(pick)];paths=[]
                    for f in seq:
                        a=indexed[f];im=images[str(a['image_id'])];b=a['bbox_image']
                        box=[max(0,math.floor(b['x'])),max(0,math.floor(b['y'])),min(im['width'],math.ceil(b['x']+b['w'])),min(im['height'],math.ceil(b['y']+b['h']))]
                        dest=out/'queries'/selected['name']/f'track_{track}_frame_{f:06d}.png'
                        paths.append(str(dest))
                        requests[im['file_name']].append((dest,box,{'path':str(dest),'pid':pid,
                            'domain':'query','split':'test','annotation_id':str(a['id']),
                            'game_id':selected['game_id'],'clip':selected['name'],'track_id':track,
                            'frame_index':f,'bbox_clamped':box,'width':box[2]-box[0],'height':box[3]-box[1]}))
                    clips.append({'query_id':f"{selected['name']}_{track}_{seq[0]}",'pid':pid,
                        'paths':paths,'game_id':selected['game_id'],'clip':selected['name'],'track_id':track,
                        'frame_indices':seq,'frame_rate':selected['frame_rate'],
                        'span_seconds':(seq[-1]-seq[0])/selected['frame_rate']})
            for filename, items in requests.items():
                with Image.open(io.BytesIO(archive.read(f"{selected['name']}/img1/{filename}"))) as source:
                    source=source.convert('RGB')
                    for dest,box,row in items:
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        source.crop(box).save(dest)
                        row['sha256']=digest_file(dest);frame_rows.append(row)
            print(f"Prepared {selected['name']} ({len(requests)} source frames)",flush=True)
    write_manifest(out/'queries.jsonl', clips)
    write_manifest(out/'query_frames.jsonl', frame_rows)
    mapping = [mapping[pid] for pid in sorted({r['pid'] for r in clips})]
    with (out/'identity_mapping_required.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
    result={'query_clips':len(clips),'distinct_query_frames':len(frame_rows),
        'anonymous_clip_team_number_identities':len(mapping),'source_matches':len(by_game),
        'source_clips':len(chosen),'exclusions':dict(exclusions),
        'query_manifest_sha256':digest_file(out/'queries.jsonl'),
        'named_identity_web_gallery_evaluated':False,
        'reason':'Name/club mapping and eight independent web images per named player still required'}
    atomic_json(report/'soccernet_preparation.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':main()
