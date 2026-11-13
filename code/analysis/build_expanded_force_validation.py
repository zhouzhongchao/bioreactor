"""Assemble all complete recorded-input evaluation rows; exclude reconstruction."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import pandas as pd
from evaluate_external_force_sets import features_from_record

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual'
OLD=ROOT/'data/exports/force_external_test_20261028'
OUT=ROOT/'data/exports/expanded_force_validation_20261028'
SEEDS=[2056,2059,2037]


def main():
    import sys
    sys.path.insert(0,str(DATA))
    from train_haw36_shared_fusion_fxyz import load_response_targets
    source=DATA/'haw60_self_attention_dataset'
    names=pd.read_csv(source/'feature_map.csv').sort_values('position_in_60D_0based').feature.tolist()
    raw=pd.read_csv(source/'haw60_matched_engineered_raw.csv').set_index('row_index',drop=False)
    scaler=pd.read_csv(source/'train_scaler_stats.csv').set_index('feature')
    mu=scaler.loc[names,'train_mean'].to_numpy();sigma=scaler.loc[names,'train_std'].to_numpy()
    times,targets,_=load_response_targets(DATA/'response.xlsx')
    train_ids=pd.read_csv(source/'train.csv').row_index.to_numpy(dtype=int)
    test_ids=set(pd.read_csv(source/'test.csv').row_index.to_numpy(dtype=int))
    assert not (set(train_ids)&test_ids)
    static_cols=[ch*9+j for ch in range(6) for j in [0,1,2,3]]+list(range(54,57))
    def fingerprint(x,y):
        a=np.round(np.r_[x[static_cols],y],6);a[np.abs(a)<.0000005]=0
        return hashlib.sha256(a.astype('<f8').tobytes()).digest()
    train_x=raw.loc[train_ids,names].to_numpy(dtype=float)
    train_fingerprints={fingerprint(x,y) for x,y in zip(train_x,targets[train_ids])}
    previous=json.loads((OLD/'external_test_results.json').read_text())
    # A candidate's inclusion depends on recorded input/target availability only.
    # Original held-out conditions + six complete other files; E07 is reconstructed.
    labels=['Original','P25']+[name for name,m in previous['datasets'].items()
                                if name.startswith('E') and 'input_reconstruction' not in m]
    assert labels==['Original','P25','E01','E02','E03','E04','E05','E06']
    cases=[]
    for label in labels:
        if label=='Original':
            ref=pd.read_csv(DATA/'dimension_original_baseline_20261028/36D/test_predictions_seed2056.csv')
            ids=ref.row_index.to_numpy(dtype=int);x=raw.loc[ids,names].to_numpy(dtype=float);y=targets[ids]
            metadata=ref[['row_index','time','process_id']].rename(columns={'time':'response_time_s'})
            origin=(DATA/'response.xlsx').relative_to(ROOT).as_posix()
            preds={}
            for dim in [12,36,42,54,60]:
                folder=DATA/('dimension_12_original_baseline_20261028' if dim==12 else 'dimension_original_baseline_20261028')/('%dD'%dim)
                seed_frames=[pd.read_csv(folder/('test_predictions_seed%d.csv'%s)) for s in SEEDS]
                for frame in seed_frames:assert np.array_equal(frame.row_index.to_numpy(),ids)
                stack=np.array([f[['Fx_pred_n','Fy_pred_n','Fz_pred_n']].to_numpy() for f in seed_frames])
                frame=metadata.copy();frame['time_s']=frame.response_time_s-frame.response_time_s.iloc[0]
                p=stack.mean(axis=0);v=stack.var(axis=0,ddof=0)
                for k,a in enumerate(['Fx','Fy','Fz']):
                    frame[a+'_true_N']=y[:,k];frame[a+'_pred_N']=p[:,k];frame[a+'_variance_N2']=v[:,k];frame[a+'_error_N']=p[:,k]-y[:,k]
                preds[dim]=frame
        else:
            preds={dim:pd.read_csv(OLD/('%s_%dD_predictions.csv'%(label,dim))) for dim in [12,36,42,54,60]}
            ref=preds[36];metadata=ref[['row_index','response_time_s','process_id']].copy()
            if label=='P25':
                ids=ref.row_index.to_numpy(dtype=int);assert set(ids)<=test_ids
                x=raw.loc[ids,names].to_numpy(dtype=float);y=targets[ids]
                origin=(DATA/'response.xlsx').relative_to(ROOT).as_posix()
            else:
                origin=previous['datasets'][label]['source']
                x,y,m,rejected=features_from_record(ROOT/origin,names,mu,sigma,return_raw=True)
                assert not rejected
                assert np.array_equal(ref.row_index.to_numpy(),[r['row_index'] for r in m])
            for frame in preds.values():
                assert np.array_equal(frame.row_index.to_numpy(),metadata.row_index.to_numpy())
                assert np.allclose(frame[['Fx_true_N','Fy_true_N','Fz_true_N']].to_numpy(),y,atol=1e-9,rtol=0)
        assert len(x)==len(y)==len(metadata) and np.isfinite(x).all() and np.isfinite(y).all()
        cases.append((label,x,y,metadata,origin,preds))
    OUT.mkdir(parents=True,exist_ok=True)
    seen={};kept_cases=[];manifest=[]
    for label,x,y,metadata,origin,preds in cases:
        keep=np.ones(len(x),dtype=bool);training_overlap=0;cross_file_duplicates=0
        for i,(xx,yy) in enumerate(zip(x,y)):
            key=fingerprint(xx,yy)
            if key in train_fingerprints:keep[i]=False;training_overlap+=1
            elif key in seen and seen[key]!=label:keep[i]=False;cross_file_duplicates+=1
            else:seen[key]=label
        manifest.append({'dataset':label,'source':origin,'rows_before':len(x),'rows_after':int(keep.sum()),
                         'exact_training_overlap_removed':training_overlap,'exact_cross_source_duplicates_removed':cross_file_duplicates})
        kept_cases.append((label,x[keep],y[keep],metadata.loc[keep].reset_index(drop=True),origin,
                           {d:f.loc[keep].reset_index(drop=True) for d,f in preds.items()}))
    all_features=[];info={'datasets':{},'results':[],'protocol':'Expanded recorded-input evaluation dataset. Original test processes 4/10/16, original held-out process25, and six other recording files. Reconstruction-derived E07 excluded. Frozen existing models/scalers; no retraining or score-based inclusion. Exact training overlaps and cross-source repeated raw states+targets removed; repeated rows within a recording retained. Files need not represent independent statistical experiments.'}
    for label,x,y,m,origin,preds in kept_cases:
        base=m.copy();base.insert(0,'dataset_id',label);base['source']=origin
        base['time_s']=base.response_time_s-base.response_time_s.iloc[0]
        for j,n in enumerate(names):base[n]=x[:,j]
        for j,a in enumerate(['Fx','Fy','Fz']):base[a+'_true_N']=y[:,j]
        all_features.append(base)
        info['datasets'][label]={'source':origin,'rows':len(x),'force_min_N':y.min(axis=0).tolist(),
            'force_max_N':y.max(axis=0).tolist(),'abs_le_1N_pct':(100*(abs(y)<=1).mean(axis=0)).tolist(),
            'abs_gt_5N_pct':(100*(abs(y)>5).mean(axis=0)).tolist()}
        for dim,frame in preds.items():
            frame.to_csv(OUT/('%s_%dD_predictions.csv'%(label,dim)),index=False)
            err=frame[['Fx_error_N','Fy_error_N','Fz_error_N']].to_numpy();rmse=np.sqrt(np.mean(err**2,axis=0))
            info['results'].append({'dataset':label,'input_dimension':dim,'n':len(x),
                'fx_rmse_N':float(rmse[0]),'fy_rmse_N':float(rmse[1]),'fz_rmse_N':float(rmse[2]),'macro_rmse_N':float(rmse.mean()),
                'zero_predictor_rmse_N':np.sqrt(np.mean(y**2,axis=0)).tolist(),
                'active_abs_gt_1N_rmse_N':[float(np.sqrt(np.mean(err[abs(y[:,k])>1,k]**2))) if np.any(abs(y[:,k])>1) else None for k in range(3)]})
    combined=pd.concat(all_features,ignore_index=True)
    combined.to_csv(OUT/'expanded_validation_features_targets.csv',index=False)
    audit={'rows_before':sum(r['rows_before'] for r in manifest),'rows_after':len(combined),
        'original_reported_test_n':3157,'sources':manifest,'feature_names':names,'force_axis_order':['Fx','Fy','Fz'],
        'usage':'Evaluation only; no fitting, no modification of deployed GUI model',
        'deduplication':'Rounded-6-decimal fingerprint of 24 raw channel P/L/active/passive values, XYZ pose and three true forces; training matches removed, cross-source matches removed, within-source repeated states retained',
        'reconstructed_input_samples_included':False}
    (OUT/'manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    (OUT/'external_test_results.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
