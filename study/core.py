"""Shared experiment recording utilities, copied with each independent repository."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def peak_memory_bytes():
    """Return the process peak in bytes when the operating system exposes it."""
    if sys.platform == "win32":
        return None
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def photos(size=192, color=False):
    from skimage import color as colors, data, transform
    camera = data.camera()
    astronaut = data.astronaut()
    if color:
        camera = np.repeat(camera[..., None], 3, axis=2)
    else:
        astronaut = colors.rgb2gray(astronaut)
    return [(name, transform.resize(a, (size,size), anti_aliasing=True))
            for name,a in [('camera',camera),('astronaut',astronaut)]]


def panel(path, entries):
    h,w=entries[0][1].shape[:2]
    canvas=Image.new('RGB',(w*len(entries),h+28),'#101827')
    d=ImageDraw.Draw(canvas)
    for i,(label,a) in enumerate(entries):
        a=np.uint8(np.clip(a,0,1)*255)
        canvas.paste(Image.fromarray(a).convert('RGB'),(i*w,28))
        d.text((i*w+5,8),label,fill='white')
    canvas.save(path)


def write_run(rows, config, out, start, limitations, status='evaluated_pilot'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not rows: raise ValueError('No results')
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    evidence={'status':status,'config':config,'python':sys.version,'platform':platform.platform(),
              'dependencies':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-image','pillow','scikit-learn','openpyxl']},
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('study').glob('*.py'))},
              'wall_seconds':time.perf_counter()-start,
              'peak_process_rss_bytes':peak_memory_bytes(),
              'memory_scope':'whole-process high-water mark, not method isolated', 'limitations':limitations}
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    keys=list(rows[0]);lines=['# Executed experiment','',f'Status: {status}. Full advanced study remains incomplete.','',
                             '| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']
    for row in rows:
        vals=[f'{v:.5g}' if isinstance(v,float) else str(v) for v in row.values()]
        lines.append('| '+' | '.join(vals)+' |')
    lines+=['','## Limits','',*['- '+v for v in limitations],'','Raw observations and compute evidence accompany this report.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(out),'rows':len(rows),'seconds':evidence['wall_seconds']}))
