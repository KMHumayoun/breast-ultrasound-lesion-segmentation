# Data, license and provenance

## Executed cohort: BrEaST

Pawłowska et al. (2024), *A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis (Breast-Lesions-USG)*, TCIA Version 1. Dataset [DOI 10.7937/9WKK-Q141](https://doi.org/10.7937/9WKK-Q141); data descriptor [DOI 10.1038/s41597-024-02984-z](https://doi.org/10.1038/s41597-024-02984-z). Data are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source authors and TCIA do not endorse this implementation.

The [official TCIA collection](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) provides the files used by study/experiment.py:

- [Images and masks ZIP](https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip), 69,862,076 bytes. SHA-256: `c32e49cdfa54042e065582d0cb35978b5fc5e5b5e461b33bf773fb50cfdad355`.
- [Clinical metadata XLSX](https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx). SHA-256: `89a8874496a6f1f93960390b963f7369b90f7e72ee08a738f2d17e37f5eff8bf`.

The complete ZIP passes CRC validation. Its 522 PNG members and all 256 unique metadata cases are checked before evaluation. Four cases lack a tumor mask. Only CaseID, Image_filename, Mask_tumor_filename and Pixel_size are loaded into the pipeline. Age, diagnosis, BI-RADS and pathology fields are not model inputs. Other-object masks are excluded from the binary tumor target.

The primary descriptor states 256 scans from 256 patients. CaseID is therefore the patient grouping unit; exact image hashes are additionally checked across splits. The split manifest and source-image hashes are in results/pilot/data_manifest.json. No raw archive or clinical workbook is uploaded.

## Geometry

The descriptor defines Pixel_size as native pixel width/height in centimeters. Every case has a positive scalar value. The pipeline interprets that as isotropic native spacing, then separately adjusts y/x spacing for the square resize and converts centimeters to millimeters. See GEOMETRY_AUDIT.md and results/neural/geometry_audit.json. Physical unit accounting does not restore lost boundary detail.

## Displayed derivatives

The image/mask comparison panels and figures/heldout_examples.png are derived from the CC BY 4.0 BrEaST inputs: grayscale conversion, resizing, reference contours and predicted-mask overlays. The source dataset/DOI/license are credited in the figure and above. The example-selection rules and inference replay are recorded in figures/evidence.json. No diagnostic claim is attached to these images.

## Pretrained model initialization

The DeepLab references use official torchvision DeepLabV3–MobileNetV3 COCO-with-VOC-category weights, not medical pretraining. The downloaded file’s hash and provenance are in results/neural/evidence.json. Original model implementations/terms are provided by [torchvision](https://github.com/pytorch/vision) under its upstream license. Downloaded initialization weights and locally trained checkpoints stay in ignored data/ and are not redistributed here.

## External cohort not evaluated

BUS-UCLM, [DOI 10.17632/7fvgj4jsp7.3](https://doi.org/10.17632/7fvgj4jsp7.3), is an external candidate, not an executed result. Download requests from the original runtime returned HTTP 403. Before any comparison, verify exact version, patient/frame grouping, mask semantics and Doppler/composite/marked-image exclusions. Do not infer healthy patients from lesion-negative frames or fabricate physical spacing when it is absent.
