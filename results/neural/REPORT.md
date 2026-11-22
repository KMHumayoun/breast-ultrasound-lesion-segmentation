# U-Net and DeepLab results

Each row shows one model at one training label budget. Dice measures how closely the predicted mask matches the expert mask. Higher is better. Each patient contributes one average Dice score across training runs.

| Training labels used | Model | Mean Dice [95% interval] | Boundary distance (mm) | Missed lesions / predictions | Time per scan (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 10% | U-Net | 0.460 [0.392, 0.532] | 8.81 | 1 / 153 | 12.93 |
| 10% | U-Net without augmentation | 0.459 [0.398, 0.522] | 10.16 | 0 / 153 | 12.11 |
| 10% | DeepLab trained from scratch | 0.119 [0.093, 0.148] | 28.10 | 0 / 153 | 18.08 |
| 10% | DeepLab with frozen backbone | 0.351 [0.262, 0.438] | 6.80 | 36 / 153 | 16.93 |
| 10% | Fine-tuned DeepLab | 0.583 [0.520, 0.644] | 4.86 | 1 / 153 | 16.65 |
| 30% | U-Net | 0.573 [0.509, 0.633] | 7.70 | 0 / 153 | 10.52 |
| 30% | U-Net without augmentation | 0.555 [0.494, 0.619] | 8.03 | 0 / 153 | 11.02 |
| 30% | DeepLab trained from scratch | 0.400 [0.347, 0.454] | 7.81 | 0 / 153 | 16.89 |
| 30% | DeepLab with frozen backbone | 0.456 [0.372, 0.532] | 5.93 | 13 / 153 | 16.94 |
| 30% | Fine-tuned DeepLab | 0.663 [0.608, 0.715] | 3.49 | 1 / 153 | 16.69 |
| 100% | U-Net | 0.685 [0.622, 0.742] | 4.74 | 0 / 153 | 12.09 |
| 100% | U-Net without augmentation | 0.679 [0.609, 0.743] | 4.40 | 0 / 153 | 11.41 |
| 100% | DeepLab trained from scratch | 0.406 [0.356, 0.458] | 7.92 | 0 / 153 | 17.20 |
| 100% | DeepLab with frozen backbone | 0.526 [0.450, 0.601] | 4.43 | 11 / 153 | 16.65 |
| 100% | Fine-tuned DeepLab | 0.706 [0.646, 0.759] | 2.54 | 2 / 153 | 17.43 |

## Reading the table

- **Training labels used** is the share of training patients whose expert masks were used to fit the model. All budgets use the same validation group.
- **Boundary distance** is the median of the finite HD95 distances. It measures how far the predicted edge lies from the expert edge in either direction. A lower value is better. Empty predictions have no edge, so read this column alongside the missed-lesion count.
- **Missed lesions** counts empty masks on scans where the expert marked a lesion. Each training run makes a prediction for the same test patients.
- **The 95% interval** comes from resampling patients in this test group. All scans came from one dataset.
- **Time per scan** measures a model prediction after warm-up, with the image already on the device.

[Measurements for each scan](metrics.csv) · [Model summaries](summary.json) · [Run details](evidence.json)
