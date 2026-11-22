"""Write the results table from the measurements saved by either experiment."""
from pathlib import Path


MODEL_NAMES = {
    "pixel_logistic": "Pixel logistic regression",
    "local_feature_forest": "Random forest",
    "unet_scratch": "U-Net",
    "unet_no_augmentation": "U-Net without augmentation",
    "deeplab_scratch": "DeepLab trained from scratch",
    "deeplab_pretrained_frozen": "DeepLab with frozen backbone",
    "deeplab_pretrained_finetune": "Fine-tuned DeepLab",
}


def write_report(out: Path, summary: list, *, neural: bool) -> None:
    title = "U-Net and DeepLab results" if neural else "Logistic regression and random forest results"
    columns = ["Training labels used", "Model", "Mean Dice [95% interval]",
               "Boundary distance (mm)", "Missed lesions / predictions"]
    if neural:
        columns.append("Time per scan (ms)")
    lines = [
        f"# {title}", "",
        "Each row shows one model at one training label budget. Dice measures how closely the predicted mask matches the expert mask. Higher is better. "
        "Each patient contributes one average Dice score across training runs.", "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * (len(columns) - 2)) + " |",
    ]
    for row in summary:
        if "patient_mean" not in row:  # Classical summaries also record validation thresholds.
            continue
        low, high = row["ci95"]
        hd = row["median_finite_hd95_mm"]
        count = row["fit_cases" if neural else "boundary_failure_denominator"]
        values = [f"{row['label_fraction']:.0%}", MODEL_NAMES[row["method"]],
                  f"{row['patient_mean']:.3f} [{low:.3f}, {high:.3f}]",
                  f"{hd:.2f}" if hd is not None else "—",
                  f"{row['empty_prediction_failures']} / {count}"]
        if neural:
            values.append(f"{row['median_device_batch1_ms']:.2f}")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "", "## Reading the table", "",
        "- **Training labels used** is the share of training patients whose expert masks were used to fit the model. All budgets use the same validation group.",
        "- **Boundary distance** is the median of the finite HD95 distances. It measures how far the predicted edge lies from the expert edge in either direction. A lower value is better. Empty predictions have no edge, so read this column alongside the missed-lesion count.",
        "- **Missed lesions** counts empty masks on scans where the expert marked a lesion. Each training run makes a prediction for the same test patients.",
        "- **The 95% interval** comes from resampling patients in this test group. All scans came from one dataset.",
    ])
    if neural:
        lines.append("- **Time per scan** measures a model prediction after warm-up, with the image already on the device.")
    lines.extend(["", "[Measurements for each scan](metrics.csv) · [Model summaries](summary.json) · [Run details](evidence.json)"])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
