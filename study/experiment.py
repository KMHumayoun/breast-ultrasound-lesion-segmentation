"""Patient-separated classical breast-ultrasound segmentation on public BrEaST stills."""
import csv
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from openpyxl import load_workbook
from scipy.ndimage import uniform_filter,sobel
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .clinical import segmentation_metrics,patient_ci,audit_groups
from .core import panel,write_run

IMAGE_URL='https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip'
META_URL='https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx'
IMAGE_SHA='c32e49cdfa54042e065582d0cb35978b5fc5e5b5e461b33bf773fb50cfdad355'
META_SHA='89a8874496a6f1f93960390b963f7369b90f7e72ee08a738f2d17e37f5eff8bf'


def scientific_json(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Unsupported scientific output type: {type(value).__name__}')


def obtain(url,sha,name,max_bytes):
    p=Path('data')/name;p.parent.mkdir(exist_ok=True)
    if not p.exists():
        # Write only after complete download and checksum validation; partial files are never accepted.
        with urllib.request.urlopen(url,timeout=300) as response:payload=response.read(max_bytes+1)
        if len(payload)>max_bytes or hashlib.sha256(payload).hexdigest()!=sha:raise ValueError('Dataset size/checksum mismatch; do not evaluate partial data')
        p.write_bytes(payload)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('Cached dataset checksum mismatch')
    return p


def preprocess(image,mask,pixel_cm,size):
    if image.shape!=mask.shape or image.ndim!=2 or pixel_cm<=0:raise ValueError('Image/mask/spacing mismatch')
    original_shape=image.shape
    image=resize(image,(size,size),order=1,anti_aliasing=True,preserve_range=True)
    mask=resize(mask.astype(np.uint8),(size,size),order=0,anti_aliasing=False,preserve_range=True)>0
    # Metadata gives isotropic native centimetres/pixel; resizing changes each axis separately.
    spacing_mm=np.array(original_shape)/size*float(pixel_cm)*10
    return image.astype(np.float32),mask,spacing_mm


def features(image):
    mean=uniform_filter(image,5);variance=np.maximum(uniform_filter(image*image,5)-mean*mean,0)
    yy,xx=np.mgrid[:image.shape[0],:image.shape[1]]
    return np.stack([image,mean,np.sqrt(variance),np.hypot(sobel(image,0),sobel(image,1))/8,
                     yy/max(image.shape[0]-1,1),xx/max(image.shape[1]-1,1)],axis=-1).reshape(-1,6).astype(np.float32)


def load_dataset(size):
    images=obtain(IMAGE_URL,IMAGE_SHA,'breast-images.zip',80*1024*1024)
    metadata=obtain(META_URL,META_SHA,'breast-metadata.xlsx',1024*1024)
    archive=zipfile.ZipFile(images)
    bad=archive.testzip()
    if bad:raise ValueError('Corrupt archive member')
    names={Path(n).name:n for n in archive.namelist() if not n.endswith('/')}
    workbook=load_workbook(metadata,read_only=True,data_only=True);rows=workbook.active.iter_rows(values_only=True);headers=next(rows)
    required=['CaseID','Image_filename','Mask_tumor_filename','Pixel_size']
    if not set(required).issubset(headers):raise ValueError('Unexpected metadata schema')
    indices={k:headers.index(k) for k in required};cases=[]
    for raw in rows:
        row={k:raw[j] for k,j in indices.items()}
        image_bytes=archive.read(names[row['Image_filename']]);native=np.asarray(Image.open(io.BytesIO(image_bytes)).convert('L'),dtype=float)/255
        mask_file=row['Mask_tumor_filename']
        if mask_file:
            mask_bytes=archive.read(names[mask_file]);mask=np.asarray(Image.open(io.BytesIO(mask_bytes)).convert('L'))>0
            mask_hash=hashlib.sha256(mask_bytes).hexdigest()
        else:mask=np.zeros(native.shape,bool);mask_hash=None
        image,resized_mask,spacing=preprocess(native,mask,float(row['Pixel_size']),size)
        cases.append({'case_id':int(row['CaseID']),'image':image,'mask':resized_mask,'spacing_mm':spacing,
                      'features':features(image),'native_shape':native.shape,'image_sha256':hashlib.sha256(image_bytes).hexdigest(),
                      'mask_sha256':mask_hash,'has_native_lesion':bool(mask.any())})
    workbook.close()
    if len(cases)!=256 or len({c['case_id'] for c in cases})!=256:raise ValueError('Expected all 256 unique BrEaST cases')
    if sum(not c['has_native_lesion'] for c in cases)!=4:raise ValueError('Unexpected normal/annotated-case count')
    if len(names)!=522:raise ValueError('Unexpected archive structure; review before use')
    return cases


def split_cases(cases,seed):
    ids=np.arange(len(cases));has_lesion=np.array([c['has_native_lesion'] for c in cases])
    train,hold=train_test_split(ids,test_size=.4,stratify=has_lesion,random_state=seed)
    validation,test=train_test_split(hold,test_size=.5,stratify=has_lesion[hold],random_state=seed)
    records=[];hash_splits={}
    for split,indices in [('train',train),('validation',validation),('test',test)]:
        for i in indices:
            c=cases[i];patient=f'BrEaST-case-{c["case_id"]}'
            records.append({'case_id':patient,'patient_id':patient,'split':split})
            hash_splits.setdefault(c['image_sha256'],set()).add(split)
    audit_groups(records)
    if any(len(s)>1 for s in hash_splits.values()):raise ValueError('Exact image duplicate crosses patient split')
    return train,validation,test,records


def sampled_pixels(cases,ids,seed):
    rng=np.random.default_rng(seed);xs=[];ys=[]
    for i in ids:
        c=cases[i];labels=c['mask'].ravel();positive=np.flatnonzero(labels);negative=np.flatnonzero(~labels)
        chosen=np.r_[rng.choice(positive,min(128,len(positive)),replace=False),rng.choice(negative,min(512,len(negative)),replace=False)]
        xs.append(c['features'][chosen]);ys.append(labels[chosen])
    return np.concatenate(xs),np.concatenate(ys)


def run(cfg,out):
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=True);cases=load_dataset(cfg['size'])
    train,validation,test,records=split_cases(cases,cfg['seed'])
    # All cases correspond to distinct patients according to the primary data descriptor.
    manifest={'dataset':'BrEaST / BREAST-LESIONS-USG','data_doi':'10.7937/9WKK-Q141','article_doi':'10.1038/s41597-024-02984-z',
              'license':'CC BY 4.0','image_archive_sha256':IMAGE_SHA,'metadata_sha256':META_SHA,
              'integrity':'complete ZIP CRC check passed; 522 image/mask PNG members; all 256 metadata cases matched',
              'grouping_basis':'Primary article states 256 scans from 256 distinct patients; CaseID used as patient group',
              'split_counts':{'train':len(train),'validation':len(validation),'test':len(test)},
              'cases':[{**r,'image_sha256':next(c['image_sha256'] for c in cases if 'BrEaST-case-'+str(c['case_id'])==r['case_id'])} for r in records],
              'metadata_fields_used':['CaseID','Image_filename','Mask_tumor_filename','Pixel_size'],
              'excluded_metadata':'Age, clinical findings, diagnoses, BIRADS and pathology labels are not loaded into model features',
              'target':'union tumor mask only; other-object annotations excluded; no diagnostic classification',
              'preprocessing':'grayscale, bilinear/antialiased image resize to square; nearest-neighbor binary mask resize; axis-specific native-cm to resized-mm spacing'}
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[];summaries=[]
    for repeat in range(cfg['repeats']):
        seed=cfg['seed']+repeat;ordered=np.random.default_rng(seed).permutation(train)
        for fraction in cfg['label_fractions']:
            selected=ordered[:max(2,int(np.ceil(len(train)*fraction)))];x,y=sampled_pixels(cases,selected,seed)
            methods={'pixel_logistic':make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=400)),
                     'local_feature_forest':RandomForestClassifier(n_estimators=60,max_depth=12,min_samples_leaf=5,random_state=seed,n_jobs=1)}
            for name,model in methods.items():
                tick=time.perf_counter();model.fit(x,y);fit_ms=(time.perf_counter()-tick)*1000
                validation_prob=[model.predict_proba(cases[i]['features'])[:,1].reshape(cases[i]['mask'].shape) for i in validation]
                thresholds=[.2,.35,.5,.65,.8];scores=[]
                for threshold in thresholds:
                    scores.append(float(np.mean([segmentation_metrics(cases[i]['mask'],p>=threshold,cases[i]['spacing_mm'])['dice'] for i,p in zip(validation,validation_prob) if cases[i]['has_native_lesion']])))
                threshold=thresholds[int(np.argmax(scores))]
                for position,i in enumerate(test):
                    c=cases[i];tick=time.perf_counter();p=model.predict_proba(c['features'])[:,1].reshape(c['mask'].shape);prediction=p>=threshold;ms=(time.perf_counter()-tick)*1000
                    m=segmentation_metrics(c['mask'],prediction,c['spacing_mm'])
                    rows.append({'repeat':repeat,'label_fraction':fraction,'labeled_train_patients':len(selected),'method':name,
                                 'public_case_id':c['case_id'],'lesion_present':c['has_native_lesion'],'threshold':threshold,
                                 'validation_lesion_dice':max(scores),'dice':m['dice'],'hd95_mm':m['hd95_spacing_units'],
                                 'absolute_area_error_mm2':m['absolute_volume_error'],
                                 'predicted_foreground_fraction':float(prediction.mean()),'fit_ms':fit_ms,'inference_ms':ms})
                    # Fixed first test case and full-label seed-zero run, not selected by success.
                    if repeat==0 and fraction==1 and position==0:
                        overlay=np.repeat(c['image'][...,None],3,axis=2);overlay[prediction]=.5*overlay[prediction]+np.array([.5,0,0])
                        panel(out/f'{name}_example.png',[('Public US still',c['image']),('Reference tumor',c['mask'].astype(float)),('Pixel score',p),('Prediction overlay',overlay)])
                summaries.append({'repeat':repeat,'label_fraction':fraction,'method':name,'selected_threshold':threshold,'validation_dice':max(scores)})
    for fraction in cfg['label_fractions']:
        for name in ['pixel_logistic','local_feature_forest']:
            selected=[r for r in rows if r['label_fraction']==fraction and r['method']==name and r['lesion_present']]
            ci=patient_ci([r['dice'] for r in selected],[r['public_case_id'] for r in selected],cfg['seed'],2000)
            finite=[r['hd95_mm'] for r in selected if np.isfinite(r['hd95_mm'])]
            summaries.append({'label_fraction':fraction,'method':name,**ci,
                              'median_finite_hd95_mm':float(np.median(finite)) if finite else None,
                              'empty_prediction_failures':sum(np.isinf(r['hd95_mm']) for r in selected),
                              'boundary_failure_denominator':len(selected),
                              'ci_scope':'Patient bootstrap of per-patient means across three fits; fixed dataset/split, not clinical population or hospital uncertainty'})
    (out/'summary.json').write_text(json.dumps(summaries,indent=2,default=scientific_json)+'\n')
    write_run(rows,cfg,out,start,['Real public de-identified BrEaST still frames; research evaluation only, no clinical validation or diagnosis.',
                                'Classical pixel/local-feature models only; no U-Net or pretrained segmenter evaluated.',
                                'One fixed patient split; metadata has no recoverable center linkage, so hospital-separated/external evaluation is unavailable here.',
                                'Square resize changes aspect ratio and resolution; physical boundary/area units account for native pixel size on each resized axis.',
                                'Label fractions count fully annotated training patients; validation patients remain fully labeled for all budgets.',
                                'Test repeats share patients; bootstrap first averages each patient across fits. Boundary failures remain infinite and are counted explicitly.',
                                'Selected pixel scores are uncalibrated foreground scores, not probabilities of malignancy.',
                                'External BUS-UCLM comparison is not run; annotation/acquisition compatibility and patient mapping must be verified first.'],status='evaluated_public_medical_baseline_pilot')
    # Replace the long observation table with a focused summary while preserving every observation in CSV.
    lines=['# BrEaST classical segmentation pilot','',
           'Status: evaluated on a real public medical dataset; advanced study and clinical validation incomplete.','',
           f'All 256 cases passed archive/schema checks. Patient split: {len(train)} training, {len(validation)} validation, {len(test)} test. Primary results use lesion-positive test patients only; normal-case false alarms remain in raw metrics.','',
           '| Labeled train fraction | Method | Mean patient Dice [95% bootstrap CI] | Median finite HD95 mm | Empty-output failures / fit-cases |',
           '|---|---|---:|---:|---:|']
    for s in summaries:
        if 'patient_mean' not in s:continue
        lo,hi=s['ci95'];hd=s['median_finite_hd95_mm']
        lines.append(f"| {s['label_fraction']:.0%} | {s['method']} | {s['patient_mean']:.3f} [{lo:.3f}, {hi:.3f}] | {hd if hd is not None else 'NA'} | {s['empty_prediction_failures']} / {s['boundary_failure_denominator']} |")
    lines+=['','The bootstrap conditions on this fixed cohort/split and averages each patient across training repeats. It does not establish hospital or clinical-population generalization. Finite HD95 medians exclude failed empty outputs only while reporting their count; they must not be read alone.',
            '', 'See metrics.csv for every patient/fit, summary.json for validation selections/interval definitions, data_manifest.json for hashes/grouping, and evidence.json for actual runtime/source versions.',
            '', '## Limits','',*['- '+v for v in json.loads((out/'evidence.json').read_text())['limitations']]]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
