# Study protocol

## Question and comparisons

Measure how lesion segmentation changes with annotation budget, model family and initialization on a fixed patient-separated BrEaST cohort. Compare pixel logistic regression, local-feature random forest, compact U-Net, the same U-Net without augmentation, scratch DeepLabV3–MobileNetV3, frozen-backbone pretrained DeepLab and fully fine-tuned pretrained DeepLab.

The classical results were already available when the neural protocol was written. The [neural-stage protocol](NEURAL_PROTOCOL.md) was fixed before its first test evaluation, but it was not externally preregistered. The later paired-difference analysis is explicitly descriptive and unadjusted for multiple comparisons.

## Inputs and split

The checksum-pinned TCIA image ZIP and workbook must pass complete integrity/schema checks: 522 PNG members, 256 unique cases, all image/tumor-mask references present, four cases without tumor masks. The data descriptor identifies one patient per case. Split by CaseID with root seed 20261122 into 153/51/52 training/validation/test patients, stratified by mask presence. Reject cross-split exact-image duplicates.

Preprocessing: grayscale [0,1], bilinear/antialiased image resize to 128×128, nearest-neighbor mask resize. Exclude other-object masks. Load only case ID, filenames and native pixel size; clinical diagnosis fields are not predictors. Convert native cm/pixel to axis-specific resized mm/pixel as documented in GEOMETRY_AUDIT.md.

## Fitting and validation

For each of three seeds, shuffle the fixed training patients once and take nested subsets of 16, 46 and 153 patients. Classical models sample foreground/background pixels from those annotated cases. Neural variants use the same patient subsets; all 51 validation patients remain fully labeled at every budget.

Neural optimization and augmentation are specified in configs/neural.json and NEURAL_PROTOCOL.md. Checkpoint selection uses lesion-positive validation Dice at 0.5, with minimum 20 epochs, maximum 60 and patience ten. Then select one threshold from 0.2/0.35/0.5/0.65/0.8 using validation Dice. No test-driven selection or adaptation.

## Evaluation units

Primary: lesion-positive test Dice, averaging each patient over fitting seeds before a 2,000-draw patient bootstrap. Secondary: spacing-aware HD95, absolute area error, empty-output failures, the mask-negative case’s foreground fraction, parameters and measured compute. Report finite HD95 with its empty-output denominator. There are 52 test patients per fitted model, not 156 independent patients after three seeds.

The post-run full-label paired comparison resamples differences between per-patient mean scores (10,000 draws). It conditions on one fixed cohort/split. Fitting seeds, images and pixels are not interchangeable units of independence.

## Known limits and recorded event

DeepLab normalization state, learning rate and parameter count differ across references; the comparison does not isolate a causal pretraining effect. MPS memory is sampled allocation, not a guaranteed peak, and process RSS is a whole-process high-water mark. Device-forward latency excludes data preparation and transfer.

One no-augmentation U-Net fit at 30% labels was temporarily suspended during browser troubleshooting. Preserve its raw timings and flag it in speed comparisons; no model or data were manually altered. Details are in results/neural/run_events.json and ANALYSIS.md.

Independent patient splits, external cohorts, annotation disagreement and acquisition-specific robustness are beyond this executed comparison. Their absence does not prevent reproducing the reported single-cohort experiment, but it limits generalization claims.
