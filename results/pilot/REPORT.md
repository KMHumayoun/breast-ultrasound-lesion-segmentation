# Logistic regression and random forest results

Each row shows one model at one training label budget. Dice measures how closely the predicted mask matches the expert mask. Higher is better. Each patient contributes one average Dice score across training runs.

| Training labels used | Model | Mean Dice [95% interval] | Boundary distance (mm) | Missed lesions / predictions |
| --- | --- | ---: | ---: | ---: |
| 10% | Pixel logistic regression | 0.226 [0.184, 0.271] | 19.35 | 0 / 153 |
| 10% | Random forest | 0.441 [0.383, 0.502] | 8.97 | 0 / 153 |
| 30% | Pixel logistic regression | 0.225 [0.184, 0.271] | 18.87 | 0 / 153 |
| 30% | Random forest | 0.480 [0.424, 0.542] | 7.32 | 0 / 153 |
| 100% | Pixel logistic regression | 0.226 [0.184, 0.272] | 18.91 | 0 / 153 |
| 100% | Random forest | 0.485 [0.428, 0.547] | 7.17 | 0 / 153 |

## Reading the table

- **Training labels used** is the share of training patients whose expert masks were used to fit the model. All budgets use the same validation group.
- **Boundary distance** is the median of the finite HD95 distances. It measures how far the predicted edge lies from the expert edge in either direction. A lower value is better. Empty predictions have no edge, so read this column alongside the missed-lesion count.
- **Missed lesions** counts empty masks on scans where the expert marked a lesion. Each training run makes a prediction for the same test patients.
- **The 95% interval** comes from resampling patients in this test group. All scans came from one dataset.

[Measurements for each scan](metrics.csv) · [Model summaries](summary.json) · [Run details](evidence.json)
