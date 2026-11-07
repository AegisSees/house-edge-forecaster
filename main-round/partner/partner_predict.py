"""Offline forecasting with joint outcomes and missing-spin marginalization."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from adaptive_model import prepare,frequency,prior_for
from joint_model import pack_joint,filter_joint
from partial_history import forecast_partial
from slot_forecasting.io import load_inputs,write_submission


def predict_component(items,component,size_model,ratios):
    missing=np.array([len(i['all_y'])<i['query'].cutoff for i in items])
    result=np.empty((len(items),5));population=component['population']
    full_items=[i for i,m in zip(items,missing) if not m]
    partial_items=[i for i,m in zip(items,missing) if m]
    if full_items:
        result[~missing]=filter_joint(pack_joint(full_items,population,size_model),population,size_model,ratios,**component['long_config'])
    if partial_items:
        result[missing]=forecast_partial(partial_items,population,size_model,ratios,**component['partial_config'])
    return result


def predict_items(items,model):
    result=np.zeros((len(items),5))
    for component in model['components']:
        result+=component['weight']*predict_component(items,component,model['size_model'],model['amount_ratios'])
    for k,item in enumerate(items):
        if not len(item['all_y']):result[k]=frequency(item,prior_for(item,model['priors']),64.,20.)
    if not np.isfinite(result).all() or np.any(result<0) or not np.allclose(result.sum(1),1,atol=1e-10,rtol=0):
        raise ValueError('Invalid probability calculation')
    return {i['query'].query_id:p for i,p in zip(items,result)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path(__file__).resolve().parent/'model.json')
    parser.add_argument('--split',choices=('test','validation'),default='test')
    args=parser.parse_args();start=time.perf_counter()
    model=json.loads(args.model.read_text());metadata=json.loads((args.data/'metadata.json').read_text())
    if model['model_version']!=3 or model['payout_order']!=[0,1,2,5,20]:raise ValueError('Unsupported model')
    if any(model[k]!=metadata[k] for k in ['round','season_id']):raise ValueError('Model/data round or season differs; refit first.')
    q,h=load_inputs(args.data/args.split)
    p=predict_items(prepare(q,h),model);write_submission(args.out,p)
    print(f'Wrote {len(p)} predictions in {time.perf_counter()-start:.3f} seconds.')


if __name__=='__main__':main()
