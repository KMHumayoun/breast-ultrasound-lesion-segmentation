"""Budgeted neural segmentation with validation-only selection and patient-level reports."""
import argparse
import csv
import gc
import hashlib
import io
import json
import platform
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
from .networks import make_model,freeze_batch_statistics,segmentation_loss
from .experiment import load_dataset,split_cases,scientific_json
from .clinical import segmentation_metrics,patient_ci
from .core import panel, peak_memory_bytes


def synchronize(device):
    if device.type=='mps':torch.mps.synchronize()


def device_info():
    return {'torch':torch.__version__,'torchvision':torchvision.__version__,'mps_built':torch.backends.mps.is_built(),
            'mps_available':torch.backends.mps.is_available(),'python':sys.version,'platform':platform.platform()}


def flip_pair(image,mask,flip):
    return (np.fliplr(image).copy(),np.fliplr(mask).copy()) if flip else (image.copy(),mask.copy())


def make_batch(cases,ids,seed,epoch,augmentation,device):
    images=[];masks=[]
    for i in ids:
        case=cases[int(i)];image,mask=case['image'],case['mask']
        if augmentation:
            rng=np.random.default_rng(seed+epoch*100003+int(i)*101)
            image,mask=flip_pair(image,mask,rng.uniform()<.5)
            image=np.clip(image**rng.uniform(.85,1.15)*(1+rng.normal(0,.035,image.shape)),0,1)
        images.append(image);masks.append(mask)
    x=torch.from_numpy(np.stack(images)[:,None].astype(np.float32)).to(device)
    y=torch.from_numpy(np.stack(masks)[:,None].astype(np.float32)).to(device)
    return x,y


def predict(model,cases,ids,batch_size,device):
    model.eval();predictions=[]
    with torch.no_grad():
        for begin in range(0,len(ids),batch_size):
            x,_=make_batch(cases,ids[begin:begin+batch_size],0,0,False,device)
            predictions.extend(torch.sigmoid(model(x))[:,0].cpu().numpy())
    return predictions


def mean_lesion_dice(cases,ids,probabilities,threshold=.5):
    scores=[]
    for i,p in zip(ids,probabilities):
        truth=cases[int(i)]['mask'];prediction=p>=threshold
        if truth.any():scores.append(2*np.sum(truth&prediction)/max(int(truth.sum()+prediction.sum()),1))
    return float(np.mean(scores))


def memory_sample(device):
    return {'sampled_live_mps_bytes':torch.mps.current_allocated_memory(),
            'sampled_driver_mps_bytes':torch.mps.driver_allocated_memory()} if device.type=='mps' else {}


def train_one(name,cases,selected,validation,seed,cfg,device,history):
    torch.manual_seed(seed);np.random.seed(seed);model=make_model(name).to(device)
    augmentation=name!='unet_no_augmentation';lr=cfg['learning_rate_unet'] if name.startswith('unet') else cfg['learning_rate_deeplab']
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    best=-1.;best_epoch=0;best_state=None;stale=0;start=time.perf_counter();sampled={}
    for epoch in range(cfg['max_epochs']):
        model.train();freeze_batch_statistics(model);order=np.random.default_rng(seed+epoch).permutation(selected);losses=[];tick=time.perf_counter()
        for begin in range(0,len(order),cfg['batch_size']):
            x,y=make_batch(cases,order[begin:begin+cfg['batch_size']],seed,epoch,augmentation,device)
            optimizer.zero_grad(set_to_none=True);loss=segmentation_loss(model(x),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss; run not evaluated')
            loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
            for k,v in memory_sample(device).items():sampled[k]=max(sampled.get(k,0),v)
        probabilities=predict(model,cases,validation,cfg['batch_size'],device)
        score=mean_lesion_dice(cases,validation,probabilities)
        synchronize(device)
        history.append({'method':name,'seed':seed,'labeled_patients':len(selected),'epoch':epoch+1,
                        'mean_training_loss':float(np.mean(losses)),'validation_dice_at_05':score,'epoch_seconds':time.perf_counter()-tick})
        print(json.dumps(history[-1]),flush=True)
        if score>best+1e-5:
            best=score;best_epoch=epoch+1;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if epoch+1>=cfg['minimum_epochs'] and stale>=cfg['patience']:break
    model.load_state_dict(best_state);model.eval();synchronize(device)
    buffer=io.BytesIO();torch.save(best_state,buffer);digest=hashlib.sha256(buffer.getbuffer()).hexdigest()
    return model,{'fit_seconds':time.perf_counter()-start,'epochs_run':epoch+1,'selected_epoch':best_epoch,
                  'validation_dice_at_05':best,'checkpoint_sha256':digest,'checkpoint_bytes':buffer.getbuffer().nbytes,
                  'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'learning_rate':lr,'augmentation':augmentation,**sampled},buffer


def probe(output):
    output.mkdir(parents=True,exist_ok=True);info=device_info();device=torch.device('mps' if info['mps_available'] else 'cpu');records=[]
    for name in ['unet_scratch','deeplab_pretrained_finetune']:
        torch.manual_seed(7);model=make_model(name).to(device);model.train();freeze_batch_statistics(model)
        x=torch.rand(4,1,128,128,device=device);y=(x>.6).float();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        # One warmup plus five measured complete optimizer steps; no clinical data read.
        segmentation_loss(model(x),y).backward();optimizer.step();optimizer.zero_grad(set_to_none=True);synchronize(device)
        start=time.perf_counter()
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True);loss=segmentation_loss(model(x),y);loss.backward();optimizer.step()
        synchronize(device)
        records.append({'model':name,'device':str(device),'mean_training_step_seconds':(time.perf_counter()-start)/5,
                        'parameters':sum(p.numel() for p in model.parameters()),**memory_sample(device)})
        print(json.dumps(records[-1]),flush=True)
        del model,optimizer,x,y;gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
    (output/'probe.json').write_text(json.dumps({'environment':info,'records':records,'scope':'generated-tensor compute probe only; no medical performance'},indent=2)+'\n')


def run(cfg,out):
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=True);info=device_info();device=torch.device('mps' if info['mps_available'] else 'cpu')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
    cases=load_dataset(cfg['size']);train,validation,test,records=split_cases(cases,cfg['seed']);rows=[];history=[];fits=[]
    for repeat in range(cfg['repeats']):
        seed=cfg['seed']+repeat;ordered=np.random.default_rng(seed).permutation(train)
        for fraction in cfg['label_fractions']:
            selected=ordered[:max(2,int(np.ceil(len(train)*fraction)))]
            for name in cfg['methods']:
                if shutil.disk_usage('.').free<384*1024*1024:raise RuntimeError('Insufficient free disk for next fit; completed fit records preserved')
                if device.type=='mps':torch.mps.empty_cache()
                print(json.dumps({'starting':name,'repeat':repeat,'label_fraction':fraction,'labeled_patients':len(selected),'device':str(device)}),flush=True)
                model,fit,checkpoint=train_one(name,cases,selected,validation,seed,cfg,device,history)
                vp=predict(model,cases,validation,cfg['batch_size'],device)
                thresholds=[.2,.35,.5,.65,.8];validation_scores=[mean_lesion_dice(cases,validation,vp,t) for t in thresholds];threshold=thresholds[int(np.argmax(validation_scores))]
                fit.update({'repeat':repeat,'method':name,'label_fraction':fraction,'labeled_patients':len(selected),'threshold':threshold,'validation_selected_dice':max(validation_scores)})
                if repeat==0 and fraction==1:
                    dest=Path('data/neural-models')/f'{name}.pt';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(checkpoint.getbuffer());fit['local_checkpoint']=str(dest)
                fits.append(fit);probabilities=predict(model,cases,test,cfg['batch_size'],device)
                warm_x,_=make_batch(cases,[int(test[0])],0,0,False,device)
                with torch.no_grad():
                    for _ in range(3):model(warm_x)
                    synchronize(device);latencies=[]
                    for _ in range(10):
                        tick=time.perf_counter();model(warm_x);synchronize(device);latencies.append((time.perf_counter()-tick)*1000)
                for position,(i,p) in enumerate(zip(test,probabilities)):
                    case=cases[int(i)];prediction=p>=threshold;m=segmentation_metrics(case['mask'],prediction,case['spacing_mm'])
                    rows.append({'repeat':repeat,'label_fraction':fraction,'labeled_train_patients':len(selected),'method':name,'public_case_id':case['case_id'],
                                 'lesion_present':case['has_native_lesion'],'threshold':threshold,'selected_epoch':fit['selected_epoch'],
                                 'dice':m['dice'],'hd95_mm':m['hd95_spacing_units'],'absolute_area_error_mm2':m['absolute_volume_error'],
                                 'foreground_fraction':float(prediction.mean()),'fit_seconds':fit['fit_seconds'],
                                 'median_device_batch1_ms':float(np.median(latencies)),**{k:v for k,v in fit.items() if k.startswith('sampled_')}})
                    if repeat==0 and fraction==1 and position==0:
                        panel(out/f'{name}_fixed_case.png',[('Public US still',case['image']),('Reference tumor',case['mask'].astype(float)),('Pixel score',p),('Prediction',prediction.astype(float))])
                # Save progress after every completed fit; interrupted work remains identifiable, never marked complete.
                write_csv(out/'metrics.csv',rows);write_csv(out/'training_history.csv',history)
                (out/'fits.json').write_text(json.dumps(fits,indent=2,default=scientific_json)+'\n')
                del model,checkpoint,warm_x,probabilities,vp;gc.collect()
                if device.type=='mps':torch.mps.empty_cache()
    summary=[]
    for fraction in cfg['label_fractions']:
        for name in cfg['methods']:
            r=[v for v in rows if v['label_fraction']==fraction and v['method']==name and v['lesion_present']]
            interval=patient_ci([v['dice'] for v in r],[v['public_case_id'] for v in r],cfg['seed'],2000)
            finite=[v['hd95_mm'] for v in r if np.isfinite(v['hd95_mm'])]
            summary.append({'method':name,'label_fraction':fraction,**interval,'median_finite_hd95_mm':float(np.median(finite)) if finite else None,
                            'empty_prediction_failures':int(sum(np.isinf(v['hd95_mm']) for v in r)),'fit_cases':len(r),
                            'median_device_batch1_ms':float(np.median([v['median_device_batch1_ms'] for v in r])),
                            'interval_scope':'patient bootstrap of per-patient means across fitting seeds; conditional on fixed cohort/split'})
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=scientific_json)+'\n')
    weights=[]
    for path in sorted(Path('data/torch-cache/checkpoints').glob('*')):
        weights.append({'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
                        'source':'torchvision DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1; official PyTorch download',
                        'pretraining':'COCO subset with Pascal VOC label categories; not medical pretraining'})
    evidence={'status':'evaluated_public_medical_neural_pilot','config':cfg,'environment':info,'device':str(device),
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('study').glob('*.py'))},
              'wall_seconds':time.perf_counter()-start,'pretrained_weight_provenance':weights,
              'split_manifest':'results/pilot/data_manifest.json; same 153/51/52 patients and root seed',
              'peak_process_rss_bytes':peak_memory_bytes(),
              'limitations':['Research evaluation on one small public still-frame cohort, not clinical validation.',
                            'No external cohort or repeated patient split yet; test cases reused across fits.',
                            'Sampled MPS allocations are not instantaneous peak GPU memory; CPU RSS is whole-process high-water.',
                            'Models share patient/image/augmentation schedules but differ in pretraining, parameter count, optimization rate and actual compute.',
                            'DeepLab BatchNorm running statistics fixed for small batches; binary head replaced. 128px resize differs from original 520px weight recipe.',
                            'Deterministic algorithms requested with warnings; MPS may still have numerical/backend nondeterminism.',
                            'No native lesion-boundary annotator-disagreement study or acquisition-domain generalization established.']}
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    lines=['# Neural breast-ultrasound segmentation pilot','', 'Status: executed research evaluation; full advanced study and clinical validation incomplete.', '',
           '| Labeled patient fraction | Method | Patient Dice [95% CI] | Median finite HD95 mm | Empty outputs / fit-cases | Batch-1 ms |',
           '|---|---|---:|---:|---:|---:|']
    for s in summary:
        lo,hi=s['ci95'];hd=s['median_finite_hd95_mm'];hdt=f'{hd:.2f}' if hd is not None else 'NA'
        lines.append(f"| {s['label_fraction']:.0%} | {s['method']} | {s['patient_mean']:.3f} [{lo:.3f}, {hi:.3f}] | {hdt} | {s['empty_prediction_failures']} / {s['fit_cases']} | {s['median_device_batch1_ms']:.2f} |")
    lines+=['','Intervals average each patient across fitting seeds before resampling patients. They condition on this fixed cohort/split. Boundary failures remain counted; finite HD95 must not be interpreted without its failure denominator.','',
            '## Limitations','',*['- '+v for v in evidence['limitations']]]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed_fits':len(fits),'rows':len(rows),'seconds':evidence['wall_seconds'],'output':str(out)}),flush=True)


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/neural.json');parser.add_argument('--output',default='results/neural');parser.add_argument('--probe',action='store_true')
    args=parser.parse_args();torch.hub.set_dir(str(Path('data/torch-cache').resolve()))
    if args.probe:probe(Path(args.output))
    else:run(json.loads(Path(args.config).read_text()),Path(args.output))


if __name__=='__main__':main()
