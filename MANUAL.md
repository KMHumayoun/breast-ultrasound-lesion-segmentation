# Install and run the project

To read the results, start with the [README](README.md). To run the code, download the repository using **Code → Download ZIP** on GitHub and extract it, or clone it with Git. Open a terminal in the folder that contains `README.md` and `requirements.lock`.

Use 64-bit Python 3.12. The first full run downloads the public BrEaST scans (about 70 MB) and their metadata. Neural training also downloads model weights and writes checkpoints, so leave several GB free. The published neural run used an Apple M1 with 8 GB of memory. Only the neural models use PyTorch.

## Windows

Use a 64-bit Windows PC with the [Python 3.12 installer](https://www.python.org/downloads/windows/) and the Python launcher. Open PowerShell in the project folder. These commands call the environment’s Python directly, so PowerShell does not need to run an activation script.

```powershell
py -3.12 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
& $python -m pip install -r requirements.lock
& $python -m pytest -q
& $python -m study --config configs/pilot.json --output results/my-classical-run
```

For U-Net and DeepLab on the CPU, run these in the same PowerShell window:

```powershell
& $python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
& $python -m study.neural --probe --output results/my-device-check
& $python -m study.neural --config configs/neural.json --output results/my-neural-run
```

## macOS

Install [Python 3.12 for macOS](https://www.python.org/downloads/macos/). Open Terminal in the project folder. The classical models use the CPU. On Apple Silicon, PyTorch uses the Mac GPU through Metal when available.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pytest -q
python -m study --config configs/pilot.json --output results/my-classical-run
```

For the neural models on Apple Silicon, use the versions from the recorded Mac run:

```bash
python -m pip install -r requirements-neural.lock
python -m study.neural --probe --output results/my-device-check
python -m study.neural --config configs/neural.json --output results/my-neural-run
```

The Mac neural environment was tested on Apple Silicon. Intel Mac users can run the classical study if the pinned packages are available for their system.

## Linux

Use Python 3.12 with the `venv` module. On Ubuntu 24.04, install both with the following command. On another Linux distribution, use its package manager to install Python 3.12 and its virtual environment support.

```bash
sudo apt install python3.12 python3.12-venv
```

In the project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pytest -q
python -m study --config configs/pilot.json --output results/my-classical-run
```

For the neural models on 64-bit x86 Linux, use the CPU wheels:

```bash
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m study.neural --probe --output results/my-device-check
python -m study.neural --config configs/neural.json --output results/my-neural-run
```

## Find your results

Open `REPORT.md` in the output folder you chose. The table summarizes the models. The accompanying `metrics.csv` contains one row per test patient for each fitted model. The device check runs a few training steps on generated images to check that PyTorch works. The full study uses the BrEaST scans.

The program checks the downloaded files before reading the scans. It keeps the data and model weights in `data/`. Use a different results folder for each new run. The fitted checkpoints in `data/neural-models/` use fixed names. Copy that folder before training again if you want to keep your previous weights.

## Make the figures

From the repository root, use the Python environment created above. On Windows, use `& $python` in place of `python` in the commands below.

```bash
python -m pip install -r requirements-plots.txt
python scripts/plot_results.py
python scripts/summarize_neural.py
```

These commands rebuild the results charts from the published measurements. To rebuild the architecture diagrams alone, run `python scripts/draw_architecture.py`.

The command `python scripts/build_review_figures.py` also rebuilds the scan examples. It requires the neural environment, source images and the three matching trained checkpoints. It checks the checkpoint hashes against the recorded run before drawing any predictions. The published examples are ready to view without downloading those files.

[Data and licenses](DATA.md) · [Model training settings](NEURAL_PROTOCOL.md) · [Checks on the published results](VALIDATION.md)
