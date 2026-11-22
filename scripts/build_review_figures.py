"""Research figures from verified BrEaST inputs and preserved completed-run checkpoints.
No training, threshold selection or model selection occurs here.
"""
from pathlib import Path
import csv,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from study.experiment import load_dataset,split_cases
from study.networks import CompactUNet,BinaryDeepLab
from study.neural import predict

OUT=Path('figures');OUT.mkdir(exist_ok=True)
COLORS=['#146c94','#bd7516','#7960a5','#008572','#c84470']
NAMES={'unet_scratch':'U-Net','unet_no_augmentation':'U-Net without augmentation','deeplab_scratch':'DeepLab, scratch',
'deeplab_pretrained_frozen':'DeepLab, frozen backbone','deeplab_pretrained_finetune':'DeepLab, fine-tuned'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
EVIDENCE={'status':'post-run descriptive figures; no training','source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':{},'checkpoint_replay':[]}


def save(fig,name,footnote):
 fig.text(.02,.015,footnote,fontsize=8,color='#475569',va='bottom')
 fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(OUT/(name+'.png'),dpi=190);fig.savefig(OUT/(name+'.svg'));plt.close(fig)


def box(ax,xy,text,width=.14,height=.17,color='#e7f1f6',fontsize=10):
 x,y=xy;ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,boxstyle='round,pad=0.012',facecolor=color,edgecolor='#537287',lw=1.2))
 ax.text(x,y,text,ha='center',va='center',fontsize=fontsize)


def arrow(ax,start,end,dashed=False,color='#546c7e'):
 ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.2,linestyle='--' if dashed else '-',color=color))


def diagrams():
 fig,ax=plt.subplots(figsize=(14,6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 box(ax,(.12,.52),'256 BrEaST patients\nImages + tumor masks\nCase ID + pixel size',.19,.23)
 box(ax,(.40,.78),'153 training patients\n16 / 46 / 153 labels\n3 fitting seeds',.23,.22)
 box(ax,(.40,.45),'51 validation patients\nCheckpoint at Dice @ 0.5\nThreshold from fixed grid',.23,.22,'#fff1dc')
 box(ax,(.72,.78),'2 classical references\n5 neural variants\n18 + 45 fitted models',.23,.22)
 box(ax,(.72,.40),'Frozen fitted models\n52 held-out test patients\n51 lesion-positive + 1 negative',.25,.22,'#e0f3eb')
 box(ax,(.72,.10),'Patient Dice + bootstrap\nPhysical boundaries + area\nEmpty masks + measured latency',.25,.16)
 arrow(ax,(.22,.59),(.27,.78));arrow(ax,(.22,.51),(.27,.45));arrow(ax,(.52,.78),(.59,.78));arrow(ax,(.72,.66),(.72,.52))
 arrow(ax,(.53,.45),(.58,.41));arrow(ax,(.72,.28),(.72,.20));ax.text(.57,.59,'No test-based selection',ha='center',fontsize=10,color='#9c4c10')
 fig.suptitle('Study flow: patient separation before model fitting',fontsize=16)
 save(fig,'study_flow','All budgets use fully annotated validation patients. The same fixed test patients are reused across fitting seeds; bootstrap units are patients.')
 fig,axes=plt.subplots(2,1,figsize=(16,7),gridspec_kw={'height_ratios':[1.8,1]})
 ax=axes[0];ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('Compact U-Net · 487,713 trainable parameters',fontsize=13)
 nodes=[(.08,.78,'128×128×16'),(.21,.55,'64×64×32'),(.34,.32,'32×32×64'),(.47,.10,'16×16×128'),(.60,.32,'32×32×64'),(.73,.55,'64×64×32'),(.86,.78,'128×128×16')]
 for x,y,t in nodes:box(ax,(x,y),t,.115,.14,fontsize=9)
 for a,b in zip(nodes,nodes[1:]):arrow(ax,(a[0]+.045,a[1]-.035),(b[0]-.045,b[1]+.035))
 for a,b in [(nodes[0],nodes[-1]),(nodes[1],nodes[-2]),(nodes[2],nodes[-3])]:arrow(ax,(a[0]+.065,a[1]),(b[0]-.065,b[1]),True,'#008572')
 ax.text(.08,.97,'1-channel input',ha='center',fontsize=9);ax.text(.86,.97,'1×1 head → 1 logit',ha='center',fontsize=9)
 ax.text(.48,.97,'Down: 2× max pool  |  Up: bilinear resize + skip concatenation',ha='center',fontsize=10)
 ax.text(.03,-.10,'Each block: two 3×3 convolutions, GroupNorm (4 groups), ReLU. Skip concatenations have 192 / 96 / 48 channels.',fontsize=9)
 ax=axes[1];ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('Binary DeepLabV3–MobileNetV3 · 11,020,337 total parameters',fontsize=13,pad=12)
 labels=['Gray 128×128\nrepeat to RGB\nImageNet normalize','MobileNetV3\nbackbone\n960×8×8','ASPP: 1×1, rates\n12 / 24 / 36, pool\nproject to 256×8×8','3×3 conv + BN\nReLU + 1×1 head\n1×8×8 logit','Bilinear upsample\n128×128 sigmoid\nvalidation threshold']
 for i,t in enumerate(labels):
  x=.1+i*.2;box(ax,(x,.45),t,.165,.54,fontsize=9)
  if i:arrow(ax,(x-.115,.45),(x-.09,.45))
 save(fig,'model_architectures','Shapes omit batch dimension and were checked with a 128×128 forward pass. DeepLab BatchNorm running statistics are fixed; its frozen-backbone variant still trains the ASPP/head (8,048,385 parameters).')


def main():
 torch.set_num_threads(2);cases=load_dataset(128);train,val,test,_=split_cases(cases,20261122)
 with Path('results/neural/metrics.csv').open() as f:rows=list(csv.DictReader(f))
 fits=json.loads(Path('results/neural/fits.json').read_text());summary=json.loads(Path('results/neural/summary.json').read_text())
 if len(fits)!=45 or len(rows)!=2340:raise RuntimeError('Completed neural run required')
 for path in ['results/neural/metrics.csv','results/neural/fits.json','results/neural/summary.json','results/pilot/data_manifest.json']:
  EVIDENCE['inputs'][path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 diagrams()
 fig,axes=plt.subplots(2,2,figsize=(12,8));split_groups=[('Train',train),('Validation',val),('Test',test)];pos=[];neg=[]
 for name,ids in split_groups:pos.append(sum(cases[i]['has_native_lesion'] for i in ids));neg.append(sum(not cases[i]['has_native_lesion'] for i in ids))
 x=np.arange(3);axes[0,0].bar(x,pos,color='#146c94',label='Lesion reference');axes[0,0].bar(x,neg,bottom=pos,color='#db9f45',label='No tumor mask')
 for i,(a,b) in enumerate(zip(pos,neg)):axes[0,0].text(i,a+b+3,f'{a}+{b}',ha='center')
 axes[0,0].set(xticks=x,xticklabels=[s[0] for s in split_groups],ylabel='Patients',ylim=(0,180),title='Patient split and reference-mask presence');axes[0,0].legend(fontsize=8)
 for (name,ids),color in zip(split_groups,COLORS):
  shape=np.array([cases[i]['native_shape'] for i in ids]);axes[0,1].scatter(shape[:,1],shape[:,0],s=13,alpha=.55,label=name,color=color)
  fraction=[cases[i]['mask'].mean()*100 for i in ids if cases[i]['has_native_lesion']];axes[1,0].hist(fraction,bins=np.arange(0,101,5),alpha=.4,label=name,color=color)
 axes[0,1].set(xlabel='Native width, pixels',ylabel='Native height, pixels',title='Native image dimensions');axes[0,1].legend(fontsize=8)
 axes[1,0].set(xlabel='Reference lesion fraction at 128×128 (%)',ylabel='Patients',title='Lesion-positive masks');axes[1,0].legend(fontsize=8)
 spacing=np.array([c['spacing_mm'] for c in cases]);axes[1,1].scatter(spacing[:,1],spacing[:,0],s=13,alpha=.5,color='#008572');axes[1,1].plot([0,.75],[0,.75],'--',color='#94a3b8',lw=1)
 axes[1,1].set(xlabel='Resized x spacing, mm/pixel',ylabel='Resized y spacing, mm/pixel',title='Axis-specific physical spacing after square resize')
 fig.suptitle('Input quality control · all 256 verified BrEaST cases',fontsize=16)
 save(fig,'cohort_qc','4 cases have no tumor mask. A negative frame is not a population specificity study. Physical spacing is read from the publisher’s Pixel_size field, not PNG DPI.')
 fig,axes=plt.subplots(1,2,figsize=(13,5));methods=list(NAMES);fractions=[.1,.3,1.];width=.24
 for k,fraction in enumerate(fractions):
  s=[next(s for s in summary if s['method']==m and s['label_fraction']==fraction) for m in methods]
  axes[0].bar(np.arange(5)+(k-1)*width,[r['empty_prediction_failures'] for r in s],width,label=f'{fraction:.0%} training labels',color=COLORS[k])
 axes[0].set(xticks=range(5),xticklabels=['U-Net','U-Net\nno aug.','DeepLab\nscratch','DeepLab\nfrozen','DeepLab\nfine-tuned'],ylabel='Empty predictions / 153 lesion-positive fit-cases',title='Failures excluded from finite HD95 summaries');axes[0].legend(fontsize=8)
 values=[]
 for method in methods:
  by_patient={}
  for r in rows:
   if r['method']==method and float(r['label_fraction'])==1 and r['lesion_present']=='True':by_patient.setdefault(r['public_case_id'],[]).append(float(r['dice']))
  values.append([np.mean(v) for v in by_patient.values()])
 axes[1].boxplot(values,positions=range(1,6),showfliers=False,widths=.5)
 rng=np.random.default_rng(42)
 for i,v in enumerate(values):axes[1].scatter(i+1+rng.uniform(-.13,.13,len(v)),v,s=13,alpha=.5,color=COLORS[i])
 axes[1].set(xticks=range(1,6),xticklabels=['U-Net','U-Net\nno aug.','DeepLab\nscratch','DeepLab\nfrozen','DeepLab\nfine-tuned'],ylabel='Patient Dice, averaged over 3 fitting seeds',ylim=(-.02,1.02),title='Full-label outcomes: one dot per test patient')
 fig.suptitle('Failure analysis and patient-level spread',fontsize=16)
 save(fig,'failure_analysis','51 lesion-positive test patients × 3 fitting seeds = 153 outcomes per method/budget. Boxes show median and quartiles; points are patient means, not independent fitting repeats.')
 # Explicit post-run selection rules: median U-Net, lowest frozen-backbone outcome, one negative patient.
 def seed0(method):return [r for r in rows if r['method']==method and r['repeat']=='0' and float(r['label_fraction'])==1 and r['lesion_present']=='True']
 ordered=sorted(seed0('unet_scratch'),key=lambda r:(float(r['dice']),int(r['public_case_id'])))
 median=ordered[len(ordered)//2];failure=min(seed0('deeplab_pretrained_frozen'),key=lambda r:(float(r['dice']),int(r['public_case_id'])))
 negative=next(i for i in test if not cases[i]['has_native_lesion']);selected_ids=[int(median['public_case_id']),int(failure['public_case_id']),cases[negative]['case_id']]
 selected=[next(i for i,c in enumerate(cases) if c['case_id']==case) for case in selected_ids]
 EVIDENCE['example_selection']={'median_unet_case':selected_ids[0],'lowest_frozen_case':selected_ids[1],'lesion_negative_case':selected_ids[2],'rule':'Post-run descriptive selection from seed0/full-label outcomes. Not used for fitting, tuning or headline metrics.'}
 device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');predictions={}
 for method in ['unet_scratch','deeplab_pretrained_finetune','deeplab_pretrained_frozen']:
  fit=next(f for f in fits if f['method']==method and f['repeat']==0 and f['label_fraction']==1)
  path=Path(fit['local_checkpoint']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if digest!=fit['checkpoint_sha256']:raise RuntimeError('Saved checkpoint changed')
  model=(CompactUNet() if method.startswith('unet') else BinaryDeepLab(False)).to(device)
  model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
  predictions[method]=[p>=fit['threshold'] for p in predict(model,cases,selected,1,device)]
  checks=[]
  for i,prediction in zip(selected,predictions[method]):
   truth=cases[i]['mask'];dice=1. if not truth.any() and not prediction.any() else float(2*np.sum(truth&prediction)/max(int(truth.sum()+prediction.sum()),1))
   saved=next(r for r in rows if r['method']==method and r['repeat']=='0' and float(r['label_fraction'])==1 and int(r['public_case_id'])==cases[i]['case_id'])
   checks.append({'public_case_id':cases[i]['case_id'],'replayed_dice':dice,'saved_dice':float(saved['dice']),'absolute_difference':abs(dice-float(saved['dice']))})
  EVIDENCE['checkpoint_replay'].append({'method':method,'checkpoint_sha256':digest,'device':str(device),'threshold':fit['threshold'],'checks':checks})
  del model
  if device.type=='mps':torch.mps.empty_cache()
 fig,axes=plt.subplots(3,5,figsize=(14,8.8));titles=['Processed image','Reference boundary','U-Net','DeepLab fine-tuned','DeepLab frozen'];selection_labels=['Median U-Net Dice','Lowest frozen Dice','Lesion-negative check']
 for r,i in enumerate(selected):
  case=cases[i]
  for c in range(5):
   ax=axes[r,c];ax.imshow(case['image'],cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
   if c==1 and case['mask'].any():ax.contour(case['mask'],levels=[.5],colors=['#00d9ae'],linewidths=1.2)
   if c>=2:
    pred=predictions[['unet_scratch','deeplab_pretrained_finetune','deeplab_pretrained_frozen'][c-2]][r]
    overlay=np.zeros((*pred.shape,4));overlay[pred]=[1,.18,.38,.45];ax.imshow(overlay)
    if case['mask'].any():ax.contour(case['mask'],levels=[.5],colors=['#00d9ae'],linewidths=1)
   if r==0:ax.set_title(titles[c],fontsize=11)
   if c==0:ax.set_ylabel(f'Case {case["case_id"]}\n{selection_labels[r]}',fontsize=9)
 fig.suptitle('Held-out examples · preserved seed-0 full-label checkpoints',fontsize=16)
 save(fig,'heldout_examples','Green: reference boundary. Rose: predicted foreground. Inputs are 128×128. Examples selected by stated descriptive rules; inference replay only. BrEaST (TCIA), CC BY 4.0, DOI 10.7937/9WKK-Q141; resized/overlaid.')
 EVIDENCE['matplotlib']=matplotlib.__version__;EVIDENCE['torch']=torch.__version__;EVIDENCE['data_attribution']='BrEaST / Pawłowska et al. (2024), TCIA DOI 10.7937/9WKK-Q141, CC BY 4.0; figures resize/overlay source inputs.'
 (OUT/'evidence.json').write_text(json.dumps(EVIDENCE,indent=2)+'\n')
 print(json.dumps({'figures':5,'example_case_ids':selected_ids,'checkpoint_replay_max_dice_difference':max(c['absolute_difference'] for r in EVIDENCE['checkpoint_replay'] for c in r['checks'])}))

if __name__=='__main__':main()
