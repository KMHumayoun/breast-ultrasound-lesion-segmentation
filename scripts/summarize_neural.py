"""Post-run patient-paired descriptive analysis and figures from preserved measurements."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/neural')
LABELS={'unet_scratch':'U-Net scratch','unet_no_augmentation':'U-Net no augmentation','deeplab_scratch':'DeepLab scratch',
        'deeplab_pretrained_frozen':'DeepLab pretrained, frozen','deeplab_pretrained_finetune':'DeepLab pretrained, fine-tuned'}


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def means(rows,method):
    result={}
    for r in rows:
        if r['method']==method and float(r['label_fraction'])==1 and r['lesion_present']=='True':
            result.setdefault(r['public_case_id'],[]).append(float(r['dice']))
    if any(len(v)!=3 for v in result.values()):raise ValueError('Expected three fits per lesion-positive patient')
    return {p:float(np.mean(v)) for p,v in result.items()}


def main():
    cfg=json.loads(Path('configs/neural.json').read_text());fits=json.loads((OUT/'fits.json').read_text())
    expected=cfg['repeats']*len(cfg['label_fractions'])*len(cfg['methods'])
    if len(fits)!=expected or not all((OUT/n).exists() for n in ['summary.json','evidence.json','REPORT.md']):raise RuntimeError('Neural experiment is incomplete')
    summary=json.loads((OUT/'summary.json').read_text());rows=read_csv(OUT/'metrics.csv');classical=read_csv(Path('results/pilot/metrics.csv'))
    all_rows=rows+classical;reference=means(all_rows,'unet_scratch');comparisons=[]
    for method in [*cfg['methods'][1:],'local_feature_forest','pixel_logistic']:
        other=means(all_rows,method)
        if reference.keys()!=other.keys():raise ValueError('Patient mismatch in paired comparison')
        differences=np.array([reference[p]-other[p] for p in sorted(reference)])
        rng=np.random.default_rng(cfg['seed']);boot=differences[rng.integers(0,len(differences),(10000,len(differences)))].mean(axis=1)
        comparisons.append({'reference':'unet_scratch','comparison':method,'label_fraction':1,'patients':len(differences),
            'mean_paired_dice_difference':float(differences.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()})
    (OUT/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    normal=[]
    for name in cfg['methods']:
        r=[v for v in rows if v['method']==name and float(v['label_fraction'])==1 and v['lesion_present']=='False']
        normal.append({'method':name,'lesion_negative_test_patients':len({v['public_case_id'] for v in r}),
            'mean_false_foreground_fraction':float(np.mean([float(v['foreground_fraction']) for v in r]))})
    (OUT/'normal_case_check.json').write_text(json.dumps(normal,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');colors=['#176b9b','#b97610','#6d52a5','#00836e','#c64c6d']
    for name,color in zip(cfg['methods'],colors):
        s=sorted([v for v in summary if v['method']==name],key=lambda r:r['label_fraction'])
        x=np.arange(len(s));y=np.array([v['patient_mean'] for v in s]);ci=np.array([v['ci95'] for v in s])
        axes[0].errorbar(x,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),marker='o',capsize=3,label=LABELS[name],color=color)
        full=s[-1];axes[1].errorbar(full['median_device_batch1_ms'],full['patient_mean'],
            yerr=[[full['patient_mean']-full['ci95'][0]],[full['ci95'][1]-full['patient_mean']]],fmt='o',capsize=3,color=color)
        offset={'unet_scratch':(6,22),'unet_no_augmentation':(6,-28)}.get(name,(5,4))
        axes[1].annotate(LABELS[name],(full['median_device_batch1_ms'],full['patient_mean']),xytext=offset,textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','color':color,'lw':.7} if name.startswith('unet') else None)
    axes[0].set(xticks=[0,1,2],xticklabels=['16 (10%)','46 (30%)','153 (100%)'],xlabel='Annotated training patients',ylabel='Lesion-positive patient Dice',ylim=(0,1),title='Label budget · 51 test patients')
    axes[0].legend(fontsize=8,loc='lower right');axes[1].set(xlabel='Warm batch-one inference, milliseconds',ylabel='Lesion-positive patient Dice',ylim=(0,1),title='Full-label performance and MPS inference cost')
    axes[1].set_xlim(left=0,right=max(v['median_device_batch1_ms'] for v in summary)*1.65)
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle('BrEaST neural segmentation · fixed 153/51/52 patient split',fontsize=14)
    fig.savefig(OUT/'neural_overview.png',dpi=180);fig.savefig(OUT/'neural_overview.svg');plt.close(fig)
    lines=['# Descriptive paired and failure analysis','',
        'This post-run analysis was specified after training began. It is descriptive, unadjusted for multiple comparisons and not a preregistered confirmatory test. No training, model, threshold or protocol was changed from its results.','',
        'Compare full-label U-Net scratch with each reference after averaging each patient across three fitting seeds. Bootstrap 51 lesion-positive patients, preserving paired patient outcomes; intervals condition on this cohort and fixed split.','',
        '| U-Net minus reference | Paired mean Dice difference | 95% patient-bootstrap interval |','|---|---:|---:|']
    for c in comparisons:lines.append(f"| {c['comparison']} | {c['mean_paired_dice_difference']:.3f} | [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] |")
    lines+=['','## Lesion-negative check','','The test split has only one lesion-negative patient. Its repeated-fit foreground fraction is a failure check, not an estimate of population specificity.','',
        '| Method | Mean false foreground fraction, full labels |','|---|---:|']
    for n in normal:lines.append(f"| {n['method']} | {n['mean_false_foreground_fraction']:.5f} |")
    lines+=['','## Interpreting DeepLab transfer','','All DeepLab variants keep BatchNorm running statistics fixed for small batches. The scratch variant therefore retains its initial running statistics, while pretrained variants retain learned ones. This bundles normalization state with transferred weights; a scratch normalization ablation would be needed to isolate a causal pretraining benefit. The comparison also uses architecture-specific learning rates and different parameter counts.','','Warm batch-one latency times only the synchronized device forward pass on an already prepared tensor, after three warmups. Image loading, resizing and input transfer are excluded; it is not end-to-end application latency.','','## Timing event','','One fit (`unet_no_augmentation`, repeat 0, label fraction 0.3) was temporarily suspended during Safari troubleshooting. The displayed elapsed-time observations imply roughly 83 seconds, not a precise stopwatch interval. Its fit/epoch wall times include the pause; retain those raw times and exclude that fit from direct training-speed comparisons. Inference latency was measured after training resumed. See run_events.json.','',
        'Finite HD95 summaries must be read with the empty-output failure denominator in REPORT.md. Foreground probabilities are segmentation scores, not malignancy probabilities.']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    inputs=[OUT/'summary.json',OUT/'metrics.csv',Path('results/pilot/metrics.csv')]
    (OUT/'analysis_evidence.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'analysis_status':'post_run_descriptive',
        'figure_scope':'Patient-bootstrap intervals conditional on one fixed cohort/split; warm MPS inference times are measured, not hardware-neutral.'},indent=2)+'\n')
    print(json.dumps({'paired_comparisons':len(comparisons),'neural_fits':len(fits),'figure':str(OUT/'neural_overview.png')}))

if __name__=='__main__':main()
