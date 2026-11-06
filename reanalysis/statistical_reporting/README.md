# Selected classes and uncertainty records

The retained CSVs give each hidden-slice selector's task class for all five seeds and its Fisher/high-loss overlap. `selected_task_class_counts.json` records the corrected source archive/member SHA256 values. The release replay command checks all 25 class choices directly against the retained per-seed blind-test JSONs and writes the class table to `build/reproduced_tables/statistical_reporting/`.

The CLIP-direction and random-direction Fisher searches select landbirds in all five seeds. High loss selects landbirds in seeds 17, 18 and 21, and waterbirds in seeds 19 and 20. Different-class slices have zero overlap by construction; the small overall overlap combines class-selection and within-class differences.

`prospective_exact_normal_summary.csv` retains full-precision per-seed arithmetic for the declared nominal normal-approximation interval convention. Hidden-slice and group-free intervals use Student-t instead, as stated in the root README and manuscript.
