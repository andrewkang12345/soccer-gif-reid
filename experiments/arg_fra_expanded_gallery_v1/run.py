"""Score the frozen expanded gallery; reuse hash-verified fixed query embeddings."""
import gc
import json
from pathlib import Path

import torch

from soccer_gif_reid.common import atomic_json, digest_file, write_manifest
from experiments.soccernet_web_gallery_v1.evaluate import extract, load_model, score, validate_inputs
from .collect import DATA, REPORT, ROOT


def main():
    torch.set_num_threads(4)
    query_path=ROOT/'data/soccernet_web_gallery_v1/arg_fra_auxiliary/four_queries.jsonl'
    queries=[json.loads(s) for s in query_path.read_text().splitlines()]
    gallery_path=DATA/'gallery.jsonl'
    gallery=[json.loads(s) for s in gallery_path.read_text().splitlines()]
    validate_inputs(queries,gallery,8)
    assert len(gallery)==168 and len({r['pid'] for r in gallery})==21 and len(queries)==109
    reviewed=json.loads((DATA/'gallery_audit.json').read_text())
    assert reviewed['gallery_manifest_sha256']==digest_file(gallery_path)
    results={}
    for label,kind,base in [('clip','clip',False),('ap3d','ap3d',False),('base','clip',True)]:
        checkpoint=ROOT/f'runs/ap3d_no_quality_v2/{kind}/best.pt'
        cache_path=ROOT/f'reports/soccernet_web_gallery_v1/arg_fra_auxiliary/{label}_four/query_features.pt'
        cached=torch.load(cache_path,map_location='cpu',weights_only=True)
        assert cached['checkpoint_sha256']==digest_file(checkpoint)
        assert cached['queries_sha256']==digest_file(query_path)
        assert cached['base']==base and cached['queries']==queries
        model,payload=load_model(checkpoint,'cuda:0')
        gf=extract(model,kind,[dict(r,paths=[r['path']]) for r in gallery],'cuda:0',base=base)
        qf=cached['features']
        metrics,predictions=score(qf,gf,queries,gallery)
        result={'model':label,'checkpoint_sha256':cached['checkpoint_sha256'],
            'query_manifest_sha256':digest_file(query_path),'gallery_manifest_sha256':digest_file(gallery_path),
            'query_features_reused':str(cache_path),'preprocessing':'Checkpoint-native letterbox; no added quality degradation',
            'reference_images_per_player':8,'reference_identities':21,'query_identities':20,
            'query_clips':109,'query_grouping':'GT identity; four adjacent sparse labeled observations; not McByte++ or dense video',
            **metrics}
        if not base:
            val=json.loads((ROOT/f'runs/ap3d_no_quality_v2/{kind}/best_metrics.json').read_text())
            test=json.loads((ROOT/f'reports/ap3d_no_quality_v2/{kind}_test_metrics.json').read_text())
            result['nba_clean_validation']={k:val['clean'][k] for k in ('rank1','mAP','queries','gallery_images','identities')}
            result['nba_clean_test']={k:test['clean'][k] for k in ('rank1','mAP','queries','gallery_images','identities')}
            result['soccer_minus_nba_validation_percentage_points']={k:100*(metrics['image_gallery'][k]-val['clean'][k]) for k in ('rank1','mAP')}
        out=REPORT/label;out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'metrics.json',result);write_manifest(out/'ranked_identities.jsonl',predictions)
        image_predictions=[];sims=qf@gf.T
        # Save all individual-photo ranks for independent AP/Rank-1 verification.
        for row,values in zip(queries,sims):
            order=torch.argsort(values,descending=True,stable=True).tolist()
            image_predictions.append({'query_id':row['query_id'],'gt_pid':row['pid'],
                'ranked_gallery':[{'pid':gallery[i]['pid'],'path':gallery[i]['path'],'similarity':float(values[i])} for i in order]})
        write_manifest(out/'ranked_images.jsonl',image_predictions)
        torch.save({'query':qf,'gallery':gf},out/'features.pt')
        results[label]=result
        print(label,{k:metrics['image_gallery'][k] for k in ('rank1','mAP')},flush=True)
        del model,payload,gf;gc.collect();torch.cuda.empty_cache()
    atomic_json(REPORT/'results.json',results)


if __name__=='__main__':main()
