# Neural breast-ultrasound segmentation pilot

Status: executed research evaluation; full advanced study and clinical validation incomplete.

| Labeled patient fraction | Method | Patient Dice [95% CI] | Median finite HD95 mm | Empty outputs / fit-cases | Batch-1 ms |
|---|---|---:|---:|---:|---:|
| 10% | unet_scratch | 0.460 [0.392, 0.532] | 8.81 | 1 / 153 | 12.93 |
| 10% | unet_no_augmentation | 0.459 [0.398, 0.522] | 10.16 | 0 / 153 | 12.11 |
| 10% | deeplab_scratch | 0.119 [0.093, 0.148] | 28.10 | 0 / 153 | 18.08 |
| 10% | deeplab_pretrained_frozen | 0.351 [0.262, 0.438] | 6.80 | 36 / 153 | 16.93 |
| 10% | deeplab_pretrained_finetune | 0.583 [0.520, 0.644] | 4.86 | 1 / 153 | 16.65 |
| 30% | unet_scratch | 0.573 [0.509, 0.633] | 7.70 | 0 / 153 | 10.52 |
| 30% | unet_no_augmentation | 0.555 [0.494, 0.619] | 8.03 | 0 / 153 | 11.02 |
| 30% | deeplab_scratch | 0.400 [0.347, 0.454] | 7.81 | 0 / 153 | 16.89 |
| 30% | deeplab_pretrained_frozen | 0.456 [0.372, 0.532] | 5.93 | 13 / 153 | 16.94 |
| 30% | deeplab_pretrained_finetune | 0.663 [0.608, 0.715] | 3.49 | 1 / 153 | 16.69 |
| 100% | unet_scratch | 0.685 [0.622, 0.742] | 4.74 | 0 / 153 | 12.09 |
| 100% | unet_no_augmentation | 0.679 [0.609, 0.743] | 4.40 | 0 / 153 | 11.41 |
| 100% | deeplab_scratch | 0.406 [0.356, 0.458] | 7.92 | 0 / 153 | 17.20 |
| 100% | deeplab_pretrained_frozen | 0.526 [0.450, 0.601] | 4.43 | 11 / 153 | 16.65 |
| 100% | deeplab_pretrained_finetune | 0.706 [0.646, 0.759] | 2.54 | 2 / 153 | 17.43 |

Intervals average each patient across fitting seeds before resampling patients. They condition on this fixed cohort/split. Boundary failures remain counted; finite HD95 must not be interpreted without its failure denominator.

## Limitations

- Research evaluation on one small public still-frame cohort, not clinical validation.
- No external cohort or repeated patient split yet; test cases reused across fits.
- Sampled MPS allocations are not instantaneous peak GPU memory; CPU RSS is whole-process high-water.
- Models share patient/image/augmentation schedules but differ in pretraining, parameter count, optimization rate and actual compute.
- DeepLab BatchNorm running statistics fixed for small batches; binary head replaced. 128px resize differs from original 520px weight recipe.
- Deterministic algorithms requested with warnings; MPS may still have numerical/backend nondeterminism.
- No native lesion-boundary annotator-disagreement study or acquisition-domain generalization established.
