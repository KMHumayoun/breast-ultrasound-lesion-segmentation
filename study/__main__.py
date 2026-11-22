import argparse
import json
from pathlib import Path
from .experiment import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/pilot.json')
    p.add_argument('--output',default='results/pilot')
    args=p.parse_args()
    run(json.loads(Path(args.config).read_text()),Path(args.output))

if __name__=='__main__': main()
