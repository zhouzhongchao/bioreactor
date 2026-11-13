"""Paired 36/42/54/60D information ablation in one fixed 60-slot network."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/output/recordings/fiting data/20260923_062542_476_slow_rom_gradual'
BASE_FEATURES = ['P_kPa', 'L_mm', 'd_active_mm', 'd_passive_mm', 'dP_dt_kPa_s', 'dL_dt_mm_s']
EXTRA_FEATURES = ['state', 'd_active_dt_mm_s', 'd_passive_dt_mm_s']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DATA/'dimension_ablation_20261028')
    parser.add_argument('--dry-run', action='store_true')
    opts = parser.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    sys.path.insert(0, str(DATA))
    import train_haw36_vib_multihead_fxyz as trainer
    from train_haw36_shared_fusion_fxyz import load_response_targets, load_split, exclude_processes
    from train_haw42_multi_encoder_fx import TrainingConfig, seed_everything

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started')
    device = torch.device('cuda')
    source = DATA/'haw60_self_attention_dataset'
    mapping = pd.read_csv(source/'feature_map.csv').sort_values('position_in_60D_0based')
    raw_columns = mapping['feature'].tolist()
    columns = [name+'_z' for name in raw_columns]
    assert len(columns)==len(set(columns))==60
    assert raw_columns[54:]==['X_deg','Y_deg','phi_deg','X_speed_deg_s','Y_speed_deg_s','phi_speed_deg_s']
    for channel in range(6):
        assert all(name.startswith('ch%d_' % (channel+1)) for name in raw_columns[channel*9:channel*9+9])
    times, targets, sheet = load_response_targets(DATA/'response.xlsx')
    indices = pd.read_csv(source/'train.csv',usecols=['row_index']).row_index.to_numpy(dtype=np.int64)
    means, stds = targets[indices].mean(axis=0), targets[indices].std(axis=0,ddof=0)
    assert np.all(stds>0)
    splits = [load_split(source/(name+'.csv'),name,columns,times,targets,means,stds)
              for name in ['train','val','test']]
    train, val, test_all = splits
    test = exclude_processes(test_all,[25])
    process_sets = [set(split.metadata.process_id.tolist()) for split in splits]
    for a in range(3):
        for b in range(a):assert not (process_sets[a]&process_sets[b])
    assert [len(s.x) for s in [train,val,test]]==[18032,4210,3157]
    assert torch.max(torch.abs(train.x.mean(dim=0))).item()<1e-4
    assert torch.max(torch.abs(train.x.std(dim=0,unbiased=False)-1)).item()<1e-4
    config = TrainingConfig()
    args = SimpleNamespace(encoder_hidden_dim=64,latent_dim=38,fusion_hidden_dims=[512,256,128],
                           head_hidden_dims=[64],attention_dim=None,attention_heads=4,attention_layers=1,
                           attention_feedforward_dim=128,initialization='default',loss_weights=[1.,1.,1.],
                           beta=1e-4,kl_warmup_epochs=20,early_stopping_metric='macro')
    original = trainer.HAW36VIBMultiHeadRegressor
    class Fixed60Regressor(original):
        mask_values = np.ones(60,dtype=np.float32)

        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            for encoder in self.encoders:
                encoder.backbone[0] = nn.Linear(9,64)
            # Identical six local encoders and Fusion in all groups. Six fixed
            # pose/speed slots enter Fusion directly; masks define availability.
            self.shared_fusion[0] = nn.Linear(6*38+6,512)
            self.register_buffer('input_feature_mask',torch.tensor(self.mask_values.copy()))

        def forward(self,x,sample=True):
            if x.ndim!=2 or x.shape[1]!=60:raise ValueError('Expected [B,60]')
            x = x*self.input_feature_mask
            latents, mus, logvars = [], [], []
            for i, encoder in enumerate(self.encoders):
                mu, logvar = encoder(x[:,i*9:(i+1)*9])
                z = mu+torch.exp(.5*logvar)*torch.randn_like(mu) if sample else mu
                latents.append(z);mus.append(mu);logvars.append(logvar)
            shared = self.shared_fusion(torch.cat(latents+[x[:,54:]],dim=1))
            return torch.cat([head(shared) for head in self.task_heads],dim=1),torch.stack(mus,dim=1),torch.stack(logvars,dim=1)

    opts.output_dir.mkdir(parents=True,exist_ok=True)
    input_hashes = {}
    for name in ['train.csv','val.csv','test.csv','feature_map.csv','train_scaler_stats.csv']:
        input_hashes[name] = hashlib.sha256((source/name).read_bytes()).hexdigest()
    protocol = {
        'variants': [36,42,54,60], 'seeds':[2025,2026,2027], 'features_all_60_slots':raw_columns,
        'architecture':'six 9->64->mu/logvar(38) VIB encoders; concatenate six standardized pose/speed slots; Fusion 234->512->256->128; three 128->64->1 heads',
        'mask_rule':'Missing standardized inputs set to zero in TRAIN/VAL/TEST; all dimensions and parameter counts unchanged',
        '36D':'6x [P,L,d_active,d_passive,dP/dt,dL/dt]',
        '42D':'36D + X,Y,phi and angular velocities',
        '54D':'6x [P,L,d_active,d_passive,state,dP/dt,dL/dt,dd_active/dt,dd_passive/dt]',
        '60D':'54D + X,Y,phi and angular velocities',
        'train_val_test_n':[len(train.x),len(val.x),len(test.x)],
        'process_ids':{'train':sorted(process_sets[0]),'val':sorted(process_sets[1]),'test':sorted(test.metadata.process_id.unique().astype(int).tolist())},
        'excluded_test_processes':[25], 'source_sha256':input_hashes,
        'hyperparameters':dict(vars(args),**vars(config)),
        'scope':'Fixed-capacity input-information ablation; NOT original deployed 36D architecture. Existing test has historical test-informed selection exposure.',
    }
    (opts.output_dir/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    seeds=[2025,2026,2027];hashes={};summary=[];tic=time.time()
    for dim in [36,42,54,60]:
        allowed = BASE_FEATURES + (EXTRA_FEATURES if dim in [54,60] else [])
        mask = np.array([float(any(n=='ch%d_%s'%(ch,f) for f in allowed for ch in range(1,7)))
                         if i<54 else float(dim in [42,60]) for i,n in enumerate(raw_columns)],dtype=np.float32)
        assert int(mask.sum())==dim
        Fixed60Regressor.mask_values = mask
        folder = opts.output_dir/('%dD'%dim);folder.mkdir(exist_ok=True)
        for seed in seeds:
            seed_everything(seed);model=Fixed60Regressor()
            digest=hashlib.sha256()
            for name,p in model.named_parameters():digest.update(name.encode());digest.update(p.detach().numpy().tobytes())
            h=digest.hexdigest()
            assert seed not in hashes or hashes[seed]==h
            hashes[seed]=h
            nparams=sum(p.numel() for p in model.parameters())
            assert nparams==342987
            x=torch.zeros(2,60);x[:,mask==0]=1000
            with torch.no_grad():assert torch.equal(model(x,sample=False)[0],model(torch.zeros_like(x),sample=False)[0])
        (folder/'input_mask.json').write_text(json.dumps({'active_dimension':dim,'mask':mask.tolist(),
            'features_used':[n for n,m in zip(raw_columns,mask) if m],
            'paired_initialization_sha256':hashes,'parameter_count':nparams},indent=2),encoding='utf8')
        print('VERIFIED',dim,'D; parameters',nparams,flush=True)
        if opts.dry_run:continue
        trainer.HAW36VIBMultiHeadRegressor=Fixed60Regressor
        results=[]
        for seed in seeds:
            metrics=trainer.train_seed(seed,args,config,train,val,test,columns,means,stds,folder,device)
            results.append(metrics)
            pd.DataFrame(results).to_csv(folder/'metrics.csv',index=False)
        row={'input_dimension':dim,'n_seeds':3,'parameter_count':nparams}
        for axis in ['fx','fy','fz','macro']:
            values=np.array([r['test_%s_rmse_n'%axis] for r in results])
            row['test_%s_rmse_mean_n'%axis]=float(values.mean())
            row['test_%s_rmse_std_n'%axis]=float(values.std(ddof=0))
        summary.append(row)
        pd.DataFrame(summary).to_csv(opts.output_dir/'summary.csv',index=False)
        (opts.output_dir/'progress.json').write_text(json.dumps({'completed_dimensions':[r['input_dimension'] for r in summary],
            'total_dimensions':4,'elapsed_s':time.time()-tic},indent=2),encoding='utf8')
    trainer.HAW36VIBMultiHeadRegressor=original
    print('COMPLETE',str(opts.output_dir),flush=True)


if __name__=='__main__':main()
