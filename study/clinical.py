"""Research evaluation helpers; these utilities do not establish clinical validity."""
from collections import defaultdict
import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt,label
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score,brier_score_loss


def audit_groups(records):
    if not records:raise ValueError('Empty cohort')
    seen=set();groups=defaultdict(set)
    for r in records:
        if any(k not in r for k in ['case_id','patient_id','split']):raise ValueError('Missing cohort field')
        if r['split'] not in ['train','validation','test','external']:raise ValueError('Invalid split')
        if r['case_id'] in seen:raise ValueError('Duplicate case')
        seen.add(r['case_id']);groups[r['patient_id']].add(r['split'])
    if any(len(v)>1 for v in groups.values()):raise ValueError('Patient overlap across splits')
    return {'cases':len(records),'patients':len(groups),'split_patient_counts':{s:sum(v=={s} for v in groups.values()) for s in ['train','validation','test','external']}}


def segmentation_metrics(truth,prediction,spacing):
    a=np.asarray(truth,dtype=bool);b=np.asarray(prediction,dtype=bool);spacing=np.asarray(spacing,dtype=float)
    if a.shape!=b.shape or len(spacing)!=a.ndim or not np.all(spacing>0):raise ValueError('Shape/spacing mismatch')
    dice=1. if not a.any() and not b.any() else 2*np.sum(a&b)/max(int(a.sum()+b.sum()),1)
    if not a.any() and not b.any():hd=0.
    elif not a.any() or not b.any():hd=float('inf')
    else:
        sa=a^binary_erosion(a);sb=b^binary_erosion(b)
        distances=np.r_[distance_transform_edt(~sb,sampling=spacing)[sa],distance_transform_edt(~sa,sampling=spacing)[sb]]
        hd=float(np.quantile(distances,.95))
    voxel=float(np.prod(spacing));return {'dice':float(dice),'hd95_spacing_units':hd,
                    'absolute_volume_error':float(abs(int(a.sum())-int(b.sum()))*voxel),
                    'signed_volume_error':float((int(b.sum())-int(a.sum()))*voxel)}


def match_lesions(truth,prediction,iou_threshold=.1):
    gt,ng=label(truth);pr,npred=label(prediction)
    if not ng or not npred:return {'true_positive':0,'false_positive':npred,'false_negative':ng}
    iou=np.array([[np.sum((gt==g)&(pr==p))/np.sum((gt==g)|(pr==p)) for p in range(1,npred+1)] for g in range(1,ng+1)])
    a,b=linear_sum_assignment(-iou);tp=int(np.sum(iou[a,b]>=iou_threshold))
    return {'true_positive':tp,'false_positive':npred-tp,'false_negative':ng-tp}


def patient_ci(values,patients,seed=0,draws=1000):
    groups=defaultdict(list)
    if len(values)!=len(patients) or not np.isfinite(values).all():raise ValueError('Invalid patient-level measurements')
    for value,patient in zip(values,patients):groups[patient].append(value)
    a=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(seed)
    if not len(a):raise ValueError('No observations')
    means=a[rng.integers(len(a),size=(draws,len(a)))].mean(axis=1)
    return {'patient_mean':float(a.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'patients':len(a)}


def binary_metrics(labels,probability):
    y=np.asarray(labels);p=np.asarray(probability);valid=np.isfinite(y)
    if not valid.any():return {'known_labels':0,'auroc':float('nan'),'brier':float('nan')}
    y=y[valid].astype(int);p=p[valid]
    if not set(y).issubset({0,1}) or np.any((p<0)|(p>1)) or not np.isfinite(p).all():raise ValueError('Invalid binary labels/probabilities')
    return {'known_labels':len(y),'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
            'brier':float(brier_score_loss(y,p))}


def phantom(seed,shape=(32,32,32),radius=4):
    rng=np.random.default_rng(seed);coordinates=np.indices(shape);center=np.array(shape)/2+rng.uniform(-3,3,len(shape))
    distance=sum((coordinates[d]-center[d])**2 for d in range(len(shape)))
    mask=distance<radius**2;image=.15+rng.normal(0,.04,shape);image[mask]+=.55
    return np.clip(image,0,1),mask
