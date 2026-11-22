# Physical geometry and metric units

BrEaST supplies physical spacing separately from its PNG images. The primary data descriptor, DOI [10.1038/s41597-024-02984-z](https://doi.org/10.1038/s41597-024-02984-z), defines `Pixel_size` as the “width and height of pixel in cm (Pixel_size)”. The actual workbook field is `Pixel_size`; all 256 cases have a positive scalar value, ranging from 0.004340278 to 0.013888889 cm per native pixel. The checksum-pinned workbook is downloaded from the official TCIA collection and remains ignored locally.

For a native image of height H and width W, resized to 128×128:

```text
spacing_y_mm = (H / 128) * Pixel_size_cm * 10
spacing_x_mm = (W / 128) * Pixel_size_cm * 10
```

`study.experiment.preprocess` computes these two values separately. `study.clinical.segmentation_metrics` passes them to SciPy's Euclidean distance transform as `sampling`, so HD95 is in millimeters. Absolute area error equals the absolute foreground-pixel-count difference times `spacing_y_mm * spacing_x_mm`, giving square millimeters. The generic function calls this `absolute_volume_error`; the two-dimensional BrEaST CSV names it `absolute_area_error_mm2` to reflect the actual dimension.

The executed resize test uses a 40×80 image with 0.01 cm native pixels resized to 20×20 and verifies spacing of 0.2×0.4 mm. This checks the cm-to-mm conversion and both resize axes. The numerical geometry audit records all-case validation, source hashes, ranges and one public example in [geometry_audit.json](results/neural/geometry_audit.json).

These units are supported by publisher metadata. They are not inferred from PNG DPI or a generic helper's ability to accept spacing. No retraining or unit correction was required. Physical accounting does not restore boundary detail lost during square resizing; nearest-neighbor mask sampling and source calibration remain limitations.
