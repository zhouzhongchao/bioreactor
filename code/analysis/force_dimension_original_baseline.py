"""Preserve deployed 36D checkpoints; extend only necessary input connections."""
import json
import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual'
MODELS=ROOT/'code/control/models/force_estimator_v1/fusion'
OUT=DATA/'dimension_original_baseline_20261028'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dimensions',nargs='+',type=int,choices=[12,36,42,54,60],default=[36,42,54,60])
    parser.add_argument('--output-dir',type=Path,default=OUT)
    options=parser.parse_args()
    output_dir=options.output_dir
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    sys.path.insert(0,str(DATA))
    import train_haw36_vib_multihead_fxyz as trainer
    from train_haw36_shared_fusion_fxyz import load_response_targets,load_split,exclude_processes,make_loader
    from train_haw42_multi_encoder_fx import TrainingConfig
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda');source=DATA/'haw60_self_attention_dataset'
    mapping=pd.read_csv(source/'feature_map.csv').sort_values('position_in_60D_0based')
    names=mapping.feature.tolist();columns=[n+'_z' for n in names]
    times,targets,_=load_response_targets(DATA/'response.xlsx')
    idx=pd.read_csv(source/'train.csv',usecols=['row_index']).row_index.to_numpy(dtype=np.int64)
    means,stds=targets[idx].mean(axis=0),targets[idx].std(axis=0,ddof=0)
    train,val,test=[load_split(source/(n+'.csv'),n,columns,times,targets,means,stds) for n in ['train','val','test']]
    test=exclude_processes(test,[25]);assert len(test.x)==3157
    config=TrainingConfig()
    args=SimpleNamespace(encoder_hidden_dim=64,latent_dim=38,fusion_hidden_dims=[512,256,128],head_hidden_dims=[64],
        attention_dim=None,attention_heads=4,attention_layers=1,attention_feedforward_dim=128,
        initialization='default',loss_weights=[1.,1.,1.],beta=1e-4,kl_warmup_epochs=20,early_stopping_metric='macro')
    original=trainer.HAW36VIBMultiHeadRegressor
    seeds=[2056,2059,2037];summary=[];ensemble=[]
    output_dir.mkdir(parents=True,exist_ok=True)
    provenance={str(seed):hashlib.sha256((MODELS/('model_seed%d.pt'%seed)).read_bytes()).hexdigest() for seed in seeds}
    protocol={'baseline':'Original deployed 36D checkpoints retained byte-for-byte; NOT retrained or replaced',
        'seeds':seeds,'hyperparameters':dict(vars(args),**vars(config)),
        'input_changes':'Only local first Linear inputs 6->2 for pressure/length only, or 6->9 for extra channel features; only Fusion first Linear inputs 228->234 when pose is present. All hidden/latent/head widths and loss unchanged.',
        'no_warm_start':'New 12/42/54/60D models trained from default initialization with the original fixed seeds; 36D uses frozen original checkpoints.',
        'selection':'No additional seed screening. Existing test has historical test-informed selection exposure.',
        'test_processes':[4,10,16],'excluded_processes':[25],'test_rows':3157,
        'original_checkpoint_sha256':provenance}
    (output_dir/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    for dim in options.dimensions:
        width=2 if dim==12 else (9 if dim in [54,60] else 6)
        pose=dim in [42,60]
        positions=[0,1] if width==2 else (list(range(9)) if width==9 else [0,1,2,3,5,6])
        indices=[ch*9+j for ch in range(6) for j in positions]
        if pose:indices+=list(range(54,60))
        assert len(indices)==dim
        class InputRegressor(original):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                if width!=6:
                    for enc in self.encoders:enc.backbone[0]=nn.Linear(width,64)
                if pose:self.shared_fusion[0]=nn.Linear(234,512)
                self.register_buffer('input_indices',torch.tensor(indices,dtype=torch.long),persistent=False)

            def forward(self,x,sample=True):
                x=x.index_select(1,self.input_indices)
                zs,ms,ls=[],[],[]
                for i,enc in enumerate(self.encoders):
                    mu,lv=enc(x[:,i*width:(i+1)*width]);ms.append(mu);ls.append(lv)
                    zs.append(mu+torch.exp(.5*lv)*torch.randn_like(mu) if sample else mu)
                if pose:zs.append(x[:,-6:])
                shared=self.shared_fusion(torch.cat(zs,dim=1))
                return torch.cat([head(shared) for head in self.task_heads],dim=1),torch.stack(ms,dim=1),torch.stack(ls,dim=1)
        folder=output_dir/('%dD'%dim);folder.mkdir(exist_ok=True)
        count=sum(p.numel() for p in InputRegressor().parameters())
        assert count=={12:337227,36:338763,42:341835,54:339915,60:342987}[dim]
        (folder/'input_config.json').write_text(json.dumps({'input_dimension':dim,'features_used':[names[i] for i in indices],
            'parameter_count':count,'channel_input_width':width,'pose_input_width':6 if pose else 0},indent=2),encoding='utf8')
        trainer.HAW36VIBMultiHeadRegressor=InputRegressor
        results=[];preds=[]
        for seed in seeds:
            if dim==36:
                src=MODELS/('model_seed%d.pt'%seed);checkpoint=torch.load(str(src),map_location=device)
                assert np.allclose(means,checkpoint['target_means_n']) and np.allclose(stds,checkpoint['target_stds_n'])
                model=InputRegressor().to(device);model.load_state_dict(checkpoint['model_state_dict'])
                loader=make_loader(test,config.batch_size,False,seed,device)
                pred,rmse,macro,kl=trainer.evaluate(model,loader,device,means,stds)
                # Check adaptation gives the identical original 36D predictions.
                reference=original().to(device);reference.load_state_dict(checkpoint['model_state_dict']);reference.eval()
                with torch.no_grad():
                    xx=test.x[:128].to(device)
                    assert torch.allclose(model(xx,sample=False)[0],reference(xx[:,indices],sample=False)[0],atol=1e-7,rtol=1e-6)
                metrics={'seed':seed,'best_epoch':checkpoint['best_epoch'],'best_val_score_n':checkpoint['best_val_score_n'],
                    'parameter_count':count,'test_macro_rmse_n':macro,'test_kl_per_latent':kl}
                for k,axis in enumerate(['fx','fy','fz']):metrics['test_%s_rmse_n'%axis]=float(rmse[k])
                frame=test.metadata.rename(columns={'response_time_s':'time'}).copy()
                for k,axis in enumerate(['x','y','z']):
                    frame['F%s_true_n'%axis]=test.y_n[:,k];frame['F%s_pred_n'%axis]=pred[:,k];frame['F%s_error_n'%axis]=pred[:,k]-test.y_n[:,k]
                frame.to_csv(folder/('test_predictions_seed%d.csv'%seed),index=False)
                shutil.copy2(str(src),str(folder/('model_seed%d.pt'%seed)))
                assert hashlib.sha256((folder/('model_seed%d.pt'%seed)).read_bytes()).hexdigest()==provenance[str(seed)]
            else:
                metrics=trainer.train_seed(seed,args,config,train,val,test,columns,means,stds,folder,device)
                frame=pd.read_csv(folder/('test_predictions_seed%d.csv'%seed))
                pred=frame[['Fx_pred_n','Fy_pred_n','Fz_pred_n']].to_numpy()
            results.append(metrics);preds.append(pred)
            pd.DataFrame(results).to_csv(folder/'metrics.csv',index=False)
        row={'input_dimension':dim,'parameter_count':count,'n_seeds':3}
        for axis in ['fx','fy','fz','macro']:
            values=np.array([r['test_%s_rmse_n'%axis] for r in results]);row['test_%s_rmse_mean_n'%axis]=values.mean();row['test_%s_rmse_std_n'%axis]=values.std(ddof=0)
            row['seed2056_%s_rmse_n'%axis]=float(results[0]['test_%s_rmse_n'%axis])
        ens=np.mean(preds,axis=0);rms=np.sqrt(np.mean((ens-test.y_n)**2,axis=0))
        for k,axis in enumerate(['fx','fy','fz']):row['ensemble_%s_rmse_n'%axis]=rms[k]
        row['ensemble_macro_rmse_n']=float(rms.mean())
        frame=test.metadata.copy()
        for k,axis in enumerate(['x','y','z']):frame['F%s_true_n'%axis]=test.y_n[:,k];frame['F%s_pred_n'%axis]=ens[:,k]
        frame.to_csv(folder/'ensemble_predictions.csv',index=False)
        summary.append(row);pd.DataFrame(summary).to_csv(output_dir/'summary.csv',index=False)
        print('COMPLETED DIMENSION',dim,json.dumps({k:float(v) for k,v in row.items()}),flush=True)
    trainer.HAW36VIBMultiHeadRegressor=original
    for seed in seeds:assert hashlib.sha256((MODELS/('model_seed%d.pt'%seed)).read_bytes()).hexdigest()==provenance[str(seed)]
    print('DONE',str(output_dir),flush=True)


if __name__=='__main__':main()
