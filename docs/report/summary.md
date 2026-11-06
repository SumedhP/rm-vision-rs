# Analysis Summary

## Per-Video Statistics

| Video | Frames | Det % | Conf % | Robot % | Updates | Median NIS | w | r | l | |h| | Fire % |
|-------|--------|-------|--------|---------|---------|------------|---|---|---|-----|---------|
| static_2m | 276 | 100.0% | 98.6% | 0.0% | 275/275 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 97.7% |
| static_2m_angled | 392 | 100.0% | 99.0% | 0.0% | 782/782 | 0.12 | 0.000 | 0.000 | 0.000 | 0.000 | 97.5% |
| static_3m | 274 | 100.0% | 98.5% | 0.0% | 273/273 | 0.01 | 0.000 | 0.000 | 0.000 | 0.000 | 97.9% |
| static_4m | 276 | 100.0% | 98.6% | 0.0% | 275/275 | 0.02 | 0.000 | 0.000 | 0.000 | 0.000 | 98.3% |
| static_5m | 275 | 100.0% | 98.5% | 0.0% | 274/274 | 0.04 | 0.000 | 0.000 | 0.000 | 0.000 | 98.5% |
| static_6m | 275 | 100.0% | 98.5% | 0.0% | 274/274 | 0.08 | 0.000 | 0.000 | 0.000 | 0.000 | 98.5% |
| spinning_2m | 950 | 100.0% | 99.2% | 60.5% | 1246/1269 | 0.86 | 14.743 | 0.247 | 0.021 | 0.095 | 56.7% |
| spinning_3m | 974 | 100.0% | 99.6% | 65.6% | 1278/1283 | 0.43 | 14.722 | 0.245 | 0.020 | 0.093 | 55.8% |
| spinning_4m | 1168 | 100.0% | 99.7% | 57.3% | 1455/1455 | 0.27 | 14.772 | 0.243 | 0.024 | 0.094 | 61.6% |
| spinning_5m | 981 | 97.3% | 96.7% | 70.6% | 1137/1137 | 0.37 | 14.861 | 0.243 | 0.022 | 0.093 | 51.1% |
| translate | 1158 | 70.6% | 69.6% | 0.0% | 815/815 | 0.08 | 0.000 | 0.000 | 0.000 | 0.000 | 69.6% |
| translate_spinning | 1259 | 67.9% | 66.6% | 72.4% | 1134/1162 | 0.84 | -11.749 | 0.242 | 0.022 | 0.093 | 22.9% |

## Replay Determinism

- **Replay files identical**: Yes (byte-identical)


The log stores JPEG frames, so replay differs slightly from the run on the raw video.

| state | cx | vx | cy | vy | cz | vz | yaw | w | r | l | h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| median abs diff (video vs replay) | 0.0046 | 0.0088 | 0.0009 | 0.0046 | 0.0005 | 0.0011 | 0.0038 | 0.0225 | 0.0002 | 0.0002 | 0.0001 |

## Prediction (figure 07)

| horizon (s) | EKF median error (cm) | hold-last-plate median error (cm) | samples |
|---|---|---|---|
| 0.05 | 6.4 | 21.6 | 3635 |
| 0.10 | 7.0 | 15.1 | 3624 |
| 0.15 | 7.6 | 21.3 | 3615 |
| 0.20 | 8.0 | 14.9 | 3614 |
| 0.30 | 8.3 | 17.5 | 3608 |

## Corner jitter (figure 03)

| video | network keypoints (px) | after refinement (px) |
|---|---|---|
| static_2m | 0.067 | 0.079 |
| static_3m | 0.112 | 0.099 |
| static_4m | 0.107 | 0.091 |
| static_5m | 0.141 | 0.086 |
| static_6m | 0.101 | 0.094 |

## Figures

- **01_detection.png**: Detection rate and median confidence per video
- **02_distance.png**: Observed plate distance vs nominal for static videos
- **03_corner_jitter.png**: Detected corner position stability with refinement ON vs OFF
- **04_spin_rate.png**: Spin rate over time, shaded for robot mode
- **05_geometry.png**: Geometric parameters (r, l, |h|) during spinning motion
- **06_nis.png**: NIS distribution of updates against chi-square(4) gate
- **07_prediction.png**: Prediction error vs horizon with baseline comparison
- **08_aimer.png**: Aimer commanded vs target yaw and acceleration limits
- **09_ballistics.png**: Flight time vs distance with vacuum ballistics curve
- **10_translate.png**: Robot center distance during translation maneuvers
- **11_latency.png**: Processing frames per second per video