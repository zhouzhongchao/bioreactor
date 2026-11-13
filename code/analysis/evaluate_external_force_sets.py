"""Evaluate existing dimension-ablation checkpoints without refitting anything."""
from pathlib import Path
import csv,json,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual'
OUT=ROOT/'data/exports/force_external_test_20261028'
LABELS=['A1+5','A2+3','A4+8','A6+7','A9','A10']
SEEDS=[2056,2059,2037]


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def features_from_record(path,names,mu,sigma,return_raw=False):
    w=None;stream=None
    if path.suffix.lower()=='.csv':
        stream=path.open(encoding='utf-8-sig',newline='');it=csv.reader(stream)
    else:
        w=load_workbook(path,read_only=True,data_only=True);it=w.worksheets[0].values
    h=list(next(it));columns={name:i for i,name in enumerate(h)}
    features=[];targets=[];meta=[];previous=None;rejected=0
    source=path.relative_to(ROOT).as_posix()
    for i,row in enumerate(it):
        try:
            t=float(row[columns['response_time_s']])
            force=np.array([float(row[columns[k]]) for k in ['F_X_N','F_Y_N','F_Z_N']])
            sensor=np.array([[float(row[columns[k]]) for k in ['P_'+lab+'_kPa','L_'+lab+'_mm','d_active_'+lab+'_mm','d_passive_'+lab+'_mm']] for lab in LABELS])
            pose=np.array([float(row[columns[k]]) for k in ['X_deg','Y_deg','phi_deg']])
            speed=np.array([float(row[columns[k]]) for k in ['X_speed_deg_s','Y_speed_deg_s','phi_speed_deg_s']])
            if not all(np.isfinite(a).all() for a in [force,sensor,pose,speed]) or not np.isfinite(t):raise ValueError('non-finite')
        except (KeyError,IndexError,TypeError,ValueError):
            rejected+=1;previous=None;continue
        rates=np.zeros_like(sensor);state=np.zeros(6)
        if previous is not None:
            pt,ps=previous;dt=t-pt
            if 1e-4<dt<=1:
                delta=sensor-ps;rates=delta/dt
                state=np.where(delta[:,0]>.25,1.,np.where(delta[:,0]<-.25,-1.,0.))
        previous=(t,sensor.copy())
        raw=[]
        for ch in range(6):raw.extend([sensor[ch,0],sensor[ch,1],sensor[ch,2],sensor[ch,3],state[ch],rates[ch,0],rates[ch,1],rates[ch,2],rates[ch,3]])
        raw.extend(pose.tolist()+speed.tolist())
        source_row=int(row[columns['source_response_row_index']]) if 'source_response_row_index' in columns else i
        features.append(raw);targets.append(force);meta.append({'row_index':source_row,'response_time_s':t,'process_id':-1,'source':source})
    if w is not None:w.close()
    if stream is not None:stream.close()
    x=np.asarray(features,dtype=np.float64)
    return (x if return_raw else ((x-mu)/sigma).astype(np.float32)),np.array(targets),meta,rejected


def main():
    torch.set_num_threads(4)
    sys.path.insert(0,str(DATA))
    from train_haw36_vib_multihead_fxyz import HAW36VIBMultiHeadRegressor
    from train_haw36_shared_fusion_fxyz import load_response_targets,load_split
    source=DATA/'haw60_self_attention_dataset'
    fmap=pd.read_csv(source/'feature_map.csv').sort_values('position_in_60D_0based')
    names=fmap.feature.tolist();assert len(names)==60
    scalers={r['feature']:r for r in read(source/'train_scaler_stats.csv')}
    mu=np.array([float(scalers[n]['train_mean']) for n in names]);sigma=np.array([float(scalers[n]['train_std']) for n in names])
    inventory=json.loads((OUT/'available_test_data.json').read_text(encoding='utf8'))
    paths=[Path(r['path']) for r in inventory if r['complete_36D_input_force_rows']>=500 and Path(r['path'])!=DATA/'response.xlsx']
    datasets=[]
    times,forces,_=load_response_targets(DATA/'response.xlsx')
    original=torch.load(str(ROOT/'code/control/models/force_estimator_v1/fusion/model_seed2056.pt'),map_location='cpu')
    ys_mu=np.array(original['target_means_n']);ys_sigma=np.array(original['target_stds_n'])
    full=load_split(source/'test.csv','test',[n+'_z' for n in names],times,forces,ys_mu,ys_sigma)
    keep=full.metadata.process_id.to_numpy()==25
    assert keep.sum()==1066
    metadata=full.metadata.loc[keep].to_dict('records')
    for r in metadata:r['source']='original held-out process 25; same recording as training, not independent recording'
    datasets.append(('P25',full.x.numpy()[keep],full.y_n[keep],metadata,0))
    for i,path in enumerate(paths,1):
        x,y,m,rejected=features_from_record(path,names,mu,sigma)
        assert len(x)>=500 and len(x)==len(y)==len(m)
        datasets.append(('E%02d'%i,x,y,m,rejected))
    metadata={}
    for name,x,y,m,rejected in datasets:
        metadata[name]={'source':m[0]['source'],'rows':len(x),'rejected_rows':rejected,
            'force_min_N':y.min(axis=0).tolist(),'force_max_N':y.max(axis=0).tolist(),
            'abs_le_1N_pct':(100*(abs(y)<=1).mean(axis=0)).tolist(),
            'abs_gt_5N_pct':(100*(abs(y)>5).mean(axis=0)).tolist(),
            'input_abs_z_max':float(np.max(abs(x))),'input_gt_5sigma_pct':float((abs(x)>5).mean()*100)}
        if name!='P25':
            match=next(r for r in inventory if Path(r['path']).resolve().relative_to(ROOT).as_posix()==m[0]['source'])
            if 'input_reconstruction' in match:metadata[name]['input_reconstruction']=match['input_reconstruction']
    (OUT/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    results=[]
    for dim in [12,36,42,54,60]:
        run=DATA/('dimension_12_original_baseline_20261028' if dim==12 else 'dimension_original_baseline_20261028')
        folder=run/('%dD'%dim)
        cfg=json.loads((folder/'input_config.json').read_text())
        indices=[names.index(n) for n in cfg['features_used']]
        width=cfg['channel_input_width'];pose=cfg['pose_input_width']>0
        class InputRegressor(HAW36VIBMultiHeadRegressor):
            def __init__(self):
                super().__init__()
                if width!=6:
                    for e in self.encoders:e.backbone[0]=nn.Linear(width,64)
                if pose:self.shared_fusion[0]=nn.Linear(234,512)

            def forward(self,x):
                x=x[:,indices];latent=[]
                for i,e in enumerate(self.encoders):latent.append(e(x[:,i*width:(i+1)*width])[0])
                if pose:latent.append(x[:,-6:])
                shared=self.shared_fusion(torch.cat(latent,dim=1))
                return torch.cat([h(shared) for h in self.task_heads],dim=1)
        models=[]
        for seed in SEEDS:
            ck=torch.load(str(folder/('model_seed%d.pt'%seed)),map_location='cpu')
            model=InputRegressor();model.load_state_dict(ck['model_state_dict']);model.eval()
            models.append((model,np.array(ck['target_means_n']),np.array(ck['target_stds_n'])))
        for label,x,y,meta,rejected in datasets:
            predictions=[]
            for model,mean,std in models:
                with torch.no_grad():
                    pred=np.concatenate([model(torch.from_numpy(x[i:i+512])).numpy() for i in range(0,len(x),512)])*std+mean
                predictions.append(pred)
            pred=np.mean(predictions,axis=0);var=np.var(predictions,axis=0,ddof=0);err=pred-y
            frame=pd.DataFrame(meta)
            frame['time_s']=frame.response_time_s-frame.response_time_s.iloc[0]
            for k,axis in enumerate(['Fx','Fy','Fz']):
                frame[axis+'_true_N']=y[:,k];frame[axis+'_pred_N']=pred[:,k];frame[axis+'_variance_N2']=var[:,k];frame[axis+'_error_N']=err[:,k]
            frame.to_csv(OUT/('%s_%dD_predictions.csv'%(label,dim)),index=False)
            rmse=np.sqrt(np.mean(err**2,axis=0));zero=np.sqrt(np.mean(y**2,axis=0))
            active=[]
            for k in range(3):
                active.append(float(np.sqrt(np.mean(err[abs(y[:,k])>1,k]**2))) if np.any(abs(y[:,k])>1) else None)
            result={'dataset':label,'input_dimension':dim,'n':len(x),'fx_rmse_N':float(rmse[0]),'fy_rmse_N':float(rmse[1]),'fz_rmse_N':float(rmse[2]),'macro_rmse_N':float(rmse.mean()),
                'zero_predictor_rmse_N':zero.tolist(),'active_abs_gt_1N_rmse_N':active}
            results.append(result);print(json.dumps(result),flush=True)
            pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in results]).to_csv(OUT/'external_test_summary.csv',index=False)
    (OUT/'external_test_results.json').write_text(json.dumps({'datasets':metadata,'results':results,
        'protocol':'Frozen model weights, original TRAIN scalers only; no fitting or target zero alignment. External causal backward differences reset at dt<=0.0001 or dt>1 s; inflate/deflate deadband deltaP +/-0.25 kPa. P25 is a held-out condition within the original recording; E01... are separate recording files. Some external files may be correlated or overlapping recordings. No model/dataset selection based on improved errors.'},indent=2),encoding='utf8')


if __name__=='__main__':main()
