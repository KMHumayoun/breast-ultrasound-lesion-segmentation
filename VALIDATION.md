# Verification record

The recorded environment is Python 3.12 on Apple M1, 8 GiB unified memory, with PyTorch 2.15.0 / torchvision 0.29.2 and verified MPS execution.

- Nine local unit checks passed with the neural dependencies installed. These cover grouping, empty masks, spacing/area semantics, patient bootstrap, resize geometry, finite network gradients, binary DeepLab output and paired image/mask flipping.
- The classical run completed 18 fits / 936 test observations. The neural run completed every configured fit: 45 unique method/budget/seed combinations, each with the same 52 test patients, totaling 2,340 observations.
- The archived source hashes refer to the code used for the published fits ([experiment revision](https://github.com/KMHumayoun/breast-ultrasound-lesion-segmentation/tree/b77f6f6251ca6161ab46aaedf9569d3ffc3d27b2)). The Windows compatibility update makes peak process memory optional. Dataset ZIP CRC, metadata schema, case count, source-image hashes and split grouping passed their checks.
- All 256 physical pixel sizes were inspected against the primary descriptor’s unit definition. The resize formula and concrete example are in GEOMETRY_AUDIT.md.
- The post-run paired analysis is descriptive, unadjusted for multiple comparisons and separated from the prospectively fixed fitting/validation choices.
- Figure-generation evidence includes input/script hashes, explicit example-selection rules, saved checkpoint hashes and replay-versus-saved scores. Figures are visually inspected before publication.

GitHub Actions defines a base job and a separate CPU PyTorch job. Hosted outcomes must be read from their actual run; local checks are not a substitute for a hosted pass. The workflow does not download a clinical cohort or rerun the full neural training.

Tests establish software contracts, not population generalization or clinical validity. Source/hash matching establishes provenance, not correctness of every scientific assumption.
