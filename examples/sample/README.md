# Sample data

A few `.npz` elevation samples produced by `scripts/build_dataset.py`, included so
that `examples/end_to_end.py` and the smoke tests run without downloading the full
TartanGround dataset (~137 GB).

Each file holds one 251x251 sample covering a 50x50 m robot-centric window at
0.2 m/cell:

| key | shape | dtype | meaning |
|---|---|---|---|
| `partial_elevation` | 251x251 | float32 | stereo elevation, NaN where unobserved |
| `partial_mask` | 251x251 | uint8 | 1 = observed by the depth cameras |
| `gt_elevation` | 251x251 | float32 | dense reference from the global point cloud |
| `gt_mask` | 251x251 | uint8 | 1 = valid ground truth |
| `frame_idx`, `reference_idx` | scalar | int32 | source frame indices |
| `bounds` | 6 | float32 | (x_min, x_max, y_min, y_max, z_min, z_max) in m |
| `resolution` | scalar | float32 | m per cell |

## License and attribution

These files are derivative works of the **TartanGround** dataset (CC BY 4.0) and
are redistributed here under the same license — see `../../LICENSE-DATA`.

> Patel, M.; Yang, F.; Qiu, Y.; Cadena, C.; Scherer, S.; Hutter, M.; Wang, W.
> "TartanGround: A Large-Scale Dataset for Ground Robot Perception and Navigation."
> IROS 2025, pp. 20524-20531. doi:10.1109/IROS60139.2025.11246002

To regenerate the full dataset from source, see `docs/DATASET.md`.
