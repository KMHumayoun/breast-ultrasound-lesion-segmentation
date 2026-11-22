# Descriptive paired and failure analysis

This post-run analysis was specified after training began. It is descriptive, unadjusted for multiple comparisons and not a preregistered confirmatory test. No training, model, threshold or protocol was changed from its results.

Compare full-label U-Net scratch with each reference after averaging each patient across three fitting seeds. Bootstrap 51 lesion-positive patients, preserving paired patient outcomes; intervals condition on this cohort and fixed split.

| U-Net minus reference | Paired mean Dice difference | 95% patient-bootstrap interval |
|---|---:|---:|
| unet_no_augmentation | 0.007 | [-0.016, 0.031] |
| deeplab_scratch | 0.279 | [0.223, 0.337] |
| deeplab_pretrained_frozen | 0.159 | [0.098, 0.225] |
| deeplab_pretrained_finetune | -0.020 | [-0.068, 0.027] |
| local_feature_forest | 0.200 | [0.146, 0.254] |
| pixel_logistic | 0.459 | [0.407, 0.509] |

## Lesion-negative check

The test split has only one lesion-negative patient. Its repeated-fit foreground fraction is a failure check, not an estimate of population specificity.

| Method | Mean false foreground fraction, full labels |
|---|---:|
| unet_scratch | 0.02948 |
| unet_no_augmentation | 0.04360 |
| deeplab_scratch | 0.08612 |
| deeplab_pretrained_frozen | 0.01062 |
| deeplab_pretrained_finetune | 0.01054 |

## Interpreting DeepLab transfer

All DeepLab variants keep BatchNorm running statistics fixed for small batches. The scratch variant therefore retains its initial running statistics, while pretrained variants retain learned ones. This bundles normalization state with transferred weights; a scratch normalization ablation would be needed to isolate a causal pretraining benefit. The comparison also uses architecture-specific learning rates and different parameter counts.

Warm batch-one latency times only the synchronized device forward pass on an already prepared tensor, after three warmups. Image loading, resizing and input transfer are excluded; it is not end-to-end application latency.

## Timing event

One fit (`unet_no_augmentation`, repeat 0, label fraction 0.3) was temporarily suspended during Safari troubleshooting. The displayed elapsed-time observations imply roughly 83 seconds, not a precise stopwatch interval. Its fit/epoch wall times include the pause; retain those raw times and exclude that fit from direct training-speed comparisons. Inference latency was measured after training resumed. See run_events.json.

Finite HD95 summaries must be read with the empty-output failure denominator in REPORT.md. Foreground probabilities are segmentation scores, not malignancy probabilities.
