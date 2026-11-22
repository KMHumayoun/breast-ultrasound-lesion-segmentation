# BrEaST classical segmentation pilot

<!-- project-reading-note -->
The completed classical stage used 153 training, 51 validation and 52 test patients. At full training labels, forest patient Dice was 0.485 [0.428, 0.547] versus 0.226 [0.184, 0.272] for pixel logistic regression. The 45-fit neural stage is still running.

Read lesion-positive Dice with empty-prediction counts and the single lesion-negative test case. Geometry comes from per-image Pixel_size metadata; see GEOMETRY_AUDIT.md.

Status: evaluated on a real public medical dataset; advanced study and clinical validation incomplete.

All 256 cases passed archive/schema checks. Patient split: 153 training, 51 validation, 52 test. Primary results use lesion-positive test patients only; normal-case false alarms remain in raw metrics.

| Labeled train fraction | Method | Mean patient Dice [95% bootstrap CI] | Median finite HD95 mm | Empty-output failures / fit-cases |
|---|---|---:|---:|---:|
| 10% | pixel_logistic | 0.226 [0.184, 0.271] | 19.34645237218662 | 0 / 153 |
| 10% | local_feature_forest | 0.441 [0.383, 0.502] | 8.967656451622304 | 0 / 153 |
| 30% | pixel_logistic | 0.225 [0.184, 0.271] | 18.87414773466226 | 0 / 153 |
| 30% | local_feature_forest | 0.480 [0.424, 0.542] | 7.324057429634175 | 0 / 153 |
| 100% | pixel_logistic | 0.226 [0.184, 0.272] | 18.91407726756815 | 0 / 153 |
| 100% | local_feature_forest | 0.485 [0.428, 0.547] | 7.166493069698295 | 0 / 153 |

The bootstrap conditions on this fixed cohort/split and averages each patient across training repeats. It does not establish hospital or clinical-population generalization. Finite HD95 medians exclude failed empty outputs only while reporting their count; they must not be read alone.

See metrics.csv for every patient/fit, summary.json for validation selections/interval definitions, data_manifest.json for hashes/grouping, and evidence.json for actual runtime/source versions.

## Limits

- Real public de-identified BrEaST still frames; research evaluation only, no clinical validation or diagnosis.
- Classical pixel/local-feature models only; no U-Net or pretrained segmenter evaluated.
- One fixed patient split; metadata has no recoverable center linkage, so hospital-separated/external evaluation is unavailable here.
- Square resize changes aspect ratio and resolution; physical boundary/area units account for native pixel size on each resized axis.
- Label fractions count fully annotated training patients; validation patients remain fully labeled for all budgets.
- Test repeats share patients; bootstrap first averages each patient across fits. Boundary failures remain infinite and are counted explicitly.
- Selected pixel scores are uncalibrated foreground scores, not probabilities of malignancy.
- External BUS-UCLM comparison is not run; annotation/acquisition compatibility and patient mapping must be verified first.
