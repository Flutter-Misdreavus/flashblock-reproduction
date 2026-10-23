#!/usr/bin/env python3
"""Independent prompt-level audit/figures; model execution is handled by DSH."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAYERS=[0,2,17,35]
HEADS=[0,8,16,24]
IDS=list(range(100,116))

def bootstrap(values):
    values=np.asarray(values,dtype=float)
    if len(values)!=16 or not np.isfinite(values).all():
        raise ValueError('Expected 16 finite prompt-level values')
    draws=np.random.default_rng(20261023).choice(values,size=(10000,16),replace=True).mean(axis=1)
    return {'n_prompts':16,'mean':float(values.mean()),'ci95':[float(x) for x in np.quantile(draws,[.025,.975])], 'values':values.tolist()}

def cell(p,metric,l,h):
    s=p['per_layer_head'][f'{l}:{h}'].get(metric)
    if not s or s.get('mean') is None:
        raise ValueError(f'Missing {metric} at {p["prompt_index"]}/{l}/{h}')
    return float(s['mean'])

def mean_prompt(p,metric):
    return float(np.mean([cell(p,metric,l,h) for l in LAYERS for h in HEADS]))

def map_metric(rows,metric):
    return np.array([[np.mean([cell(p,metric,l,h) for p in rows]) for h in HEADS] for l in LAYERS])

def layer_metric(rows,metric):
    return np.array([np.mean([np.mean([cell(p,metric,l,h) for h in HEADS]) for p in rows]) for l in LAYERS])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    src=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data=json.loads(src.read_text());group={}
    for p in data: group.setdefault(p['condition'],[]).append(p)
    if set(group)!={'dense','s2'}: raise ValueError(f'Unexpected conditions {list(group)}')
    audit={'analysis_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'aggregation':'equal weight prompts, selected layers and selected heads; 10000 prompt bootstrap draws seed20261023','versions':{'numpy':np.__version__,'matplotlib':matplotlib.__version__},'conditions':{}}
    for cond,rows in group.items():
        rows.sort(key=lambda p:p['prompt_index'])
        if [p['prompt_index'] for p in rows]!=IDS: raise ValueError('Missing/duplicate/faulty prompt IDs')
        gaps=[]
        for p in rows:
            o=p['output']
            if not 0<o['new_tokens']<=512 or not o['ids_in_vocab'] or o['contains_mask_token'] or not o['text_nonempty']:
                raise ValueError('Invalid output')
            gap=mean_prompt(p,'gap')
            if not np.isclose(gap,p['mean_gap'],rtol=0,atol=1e-9): raise ValueError('Aggregation mismatch')
            if not np.isclose(gap,mean_prompt(p,'cos_out')-mean_prompt(p,'cos_in'),rtol=0,atol=1e-6): raise ValueError('Cosine difference mismatch')
            gaps.append(gap)
        metrics={'paired_cosine_gap':bootstrap(gaps),'cosine_out':bootstrap([mean_prompt(p,'cos_out') for p in rows]),'cosine_in':bootstrap([mean_prompt(p,'cos_in') for p in rows]),'lse_out_absolute_drift':bootstrap([mean_prompt(p,'lse_out_drift') for p in rows]),'lse_in_absolute_drift':bootstrap([mean_prompt(p,'lse_in_drift') for p in rows]),'fresh_output_relative_l2':bootstrap([mean_prompt(p,'actual_full_refresh') for p in rows]),'generated_tokens':[p['output']['new_tokens'] for p in rows]}
        if cond=='s2':
            for key in ['actual_full_reuse','stale_merge_rel','stale_vs_actual']:
                metrics[key]=bootstrap([mean_prompt(p,key) for p in rows])
            totals={k:sum(p['steps'].get(k,0) for p in rows) for k in ['reuse_steps','denoise_total','denoise_steps','saving_steps','al_store_steps','refresh_steps']}
            metrics['steps']=totals
            metrics['reuse_fraction_all_steps']=totals['reuse_steps']/totals['denoise_total']
            metrics['reuse_fraction_excluding_saving']=totals['reuse_steps']/totals['denoise_steps']
            metrics['reuse_denominators_match']=all(p['per_layer_head'][f'{l}:{h}']['stale_vs_actual']['n']==p['per_layer_head'][f'{l}:{h}']['actual_full_reuse']['n'] for p in rows for l in LAYERS for h in HEADS)
            if not metrics['reuse_denominators_match']: raise ValueError('Reuse denominator mismatch')
        audit['conditions'][cond]=metrics
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    dense=group['dense'];s2=group['s2']
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),gridspec_kw={'width_ratios':[1,1,1.2]},layout='constrained')
    for ax,key,title in zip(axes[:2],['cos_out','cos_in'],['External attention','Internal attention']):
        ar=map_metric(dense,key);im=ax.imshow(ar,vmin=-1,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(4),HEADS);ax.set_yticks(range(4),LAYERS);ax.set(xlabel='Head',ylabel='Layer',title=title)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{ar[i,j]:.3f}',ha='center',va='center',fontsize=8,color='black' if ar[i,j]>.5 else 'white')
    fig.colorbar(im,ax=axes[:2],shrink=.8,label='Adjacent-step cosine (same scale)')
    vals=audit['conditions']['dense']['paired_cosine_gap'];ax=axes[2]
    ax.scatter(IDS,vals['values'],s=22,color='#246a91',label='One point per prompt')
    ax.axhline(0,color='gray',lw=.8);ax.axhline(vals['mean'],color='#c65a36',label='Prompt mean')
    ax.axhspan(*vals['ci95'],color='#c65a36',alpha=.15,label='95% prompt-bootstrap CI')
    ax.set(xlabel='GSM8K train index',ylabel='External minus internal cosine',title='Paired stability difference');ax.legend(fontsize=7)
    fig.suptitle('FlashBlock mechanism: TraDo-4B, 16 fixed prompts, dense trajectory')
    for ext in ['png','pdf']:fig.savefig(out/f'mechanism-stability.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained');x=np.arange(4);w=.35
    ax=axes[0]
    for off,key,label,color in [(-w/2,'lse_out_drift','External','#246a91'),(w/2,'lse_in_drift','Internal','#c65a36')]:
        ax.bar(x+off,layer_metric(dense,key),w,label=label,color=color)
    ax.set(ylabel='Mean absolute LSE drift',title='Partition-mass drift (dense)');ax.legend(fontsize=8)
    ax=axes[1];ax.bar(x,layer_metric(s2,'actual_full_reuse'),color='#c65a36');ax.set(ylabel='Relative L2 error',title='Actual reuse vs fresh full attention')
    ax=axes[2]
    for off,rows,key,label,color in [(-w/2,dense,'actual_full_refresh','Dense vs FP32','#246a91'),(w/2,s2,'stale_vs_actual','Reuse vs stale composition','#53a884')]:
        ax.bar(x+off,layer_metric(rows,key),w,label=label,color=color)
    ax.set(ylabel='Relative L2 residual',title='Numerical controls');ax.legend(fontsize=7)
    for ax in axes:ax.set_xticks(x,LAYERS);ax.set_xlabel('Layer')
    fig.suptitle('Prompt-weighted means; approximation error is distinct from kernel error')
    for ext in ['png','pdf']:fig.savefig(out/f'mechanism-errors.{ext}',dpi=180)
    plt.close(fig)
    (out/'independent-review.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({c:{'gap':v['paired_cosine_gap']['mean'],'ci95':v['paired_cosine_gap']['ci95']} for c,v in audit['conditions'].items()}))

if __name__=='__main__':main()
