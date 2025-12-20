# Datasets

This folder stores NPZ files for experiments. Each dataset must document its
keys, shapes, dtypes, and meaning before it is used in training or evaluation.

## NPZ Schema Template

- file: <name>.npz
- description: <short description of how the dataset was collected>
- total_frames: <int>
- time_step_sec: <float, sampling interval>

### Keys

| key | shape | dtype | description | note |
| --- | ----- | ----- | ----------- | ---- |
| obs_rgb | (T, H, W, 3) | uint8 | RGB image sequence | values in [0, 255] |
| obs_depth | (T, H, W) | float32 | Depth image sequence | meters |
| action | (T, A) | float32 | Action vector | normalized to [-1, 1] |
| reward | (T,) | float32 | Reward per step | optional |
| done | (T,) | bool | Episode termination flags | optional |

### Notes

- T is the time dimension (number of frames).
- H and W must be consistent across all samples.
- If a key is optional, explicitly state when it is missing.
