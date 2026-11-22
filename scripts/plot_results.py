from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pilot'
rows=list(csv.DictReader((OUT/'metrics.csv').open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True,'font.family':'DejaVu Sans'})
COLORS=['#1878bd','#17a58a','#d88a28','#9a65ad','#6a7786']
def mean(values):return float(np.mean(list(values)))
def finish(fig,title,footnote):
    fig.suptitle(title,fontweight='bold',fontsize=14)
    fig.text(.02,.015,footnote,fontsize=8,color='#4b5563')
    fig.tight_layout(rect=[0,.07,1,.94])
    fig.savefig(OUT/'overview.png',dpi=180,facecolor='white')
    fig.savefig(OUT/'overview.svg',facecolor='white')
    evidence={'input_sha256':hashlib.sha256((OUT/'metrics.csv').read_bytes()).hexdigest(),'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,'note':'Presentation derived from saved observations; no model fitting or test-based selection performed.'}
    (OUT/'plot_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')

summary=[s for s in json.loads((OUT/'summary.json').read_text()) if 'patient_mean' in s];fig,ax=plt.subplots(figsize=(9,4.8))
for color,method in zip(COLORS,['pixel_logistic','local_feature_forest']):
    r=[q for q in summary if q['method']==method];x=np.array([q['label_fraction'] for q in r])*100;v=np.array([q['patient_mean'] for q in r]);lo=np.array([q['ci95'][0] for q in r]);hi=np.array([q['ci95'][1] for q in r]);ax.errorbar(x,v,yerr=[v-lo,hi-v],fmt='o-',capsize=4,color=color,label=method)
ax.set(xlabel='Fully annotated training-patient fraction (%)',ylabel='Mean lesion-positive patient Dice',ylim=(0,1));ax.set_xticks([10,30,100]);ax.legend()
finish(fig,'BrEaST: classical segmentation and label budgets','153/51/52 patient split. Patient-bootstrap CIs average three fits per patient; fixed cohort/split, no clinical or external validation.')
