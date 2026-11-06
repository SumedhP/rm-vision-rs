## Ballistics: pitch to hit a target, 23 m/s, k = 0.019, ranges 2-8 m, heights -0.4/0/+0.3 m

Miss = vertical miss at the target when the solved pitch is flown by a 1e-12 Dopri5 reference.

| Method | µs per solve | worst miss mm | median miss mm |
|---|---|---|---|
| ours: RK4 5 ms + secant from vacuum pitch | 19.7 | 0.374 | 0.031 |
| ode_solvers Dopri5 (tol 1e-8) + roots Brent | 27.6 | 0.037 | 0.002 |
| ode_solvers Dopri5 (tol 1e-8) + roots secant from vacuum pitch | 10.0 | 0.050 | 0.002 |
| ode_solvers RK4 (0.1 m steps) + roots secant from vacuum pitch | 30.8 | 0.348 | 0.039 |
| our integrator + argmin BrentRoot | 46.9 | 0.390 | 0.034 |
| awakening closed form (no simulation) | 0.0 | 39.872 | 12.408 |

## Gimbal yaw trajectory: closed loop at 200 Hz against the recorded aim points

Gimbal = ideal double integrator, |acc| <= 60 rad/s², |rate| <= 10 rad/s. MPC horizon 100 x 5 ms, and it is given the *recorded* future aim points (perfect foresight). In the pipeline that future would come from the robot model.

| Video | Planner | µs per tick, median | p99 | RMS error mrad | ticks within fire tolerance |
|---|---|---|---|---|---|
| translate_spinning | ours: reactive smoother (bandwidth 60) | 0.1 | 0.2 | 12.86 | 89.3% |
| translate_spinning | tinympc-rs (sp_vision's MPC: Q 9e6, R 1, rho 1, 10 iterations) | 26.8 | 48.1 | 7.74 | 94.6% |
| translate_spinning | clarabel (same MPC as a sparse QP, interior point) | 2131.5 | 3586.2 | 7.75 | 94.7% |
| translate_spinning | osqp (same MPC as a sparse QP, ADMM, warm-started, eps 1e-4) | 2718.7 | 16681.9 | 7.90 | 94.9% |
| spinning_3m | ours: reactive smoother (bandwidth 60) | 0.1 | 0.2 | 7.80 | 97.3% |
| spinning_3m | tinympc-rs (sp_vision's MPC: Q 9e6, R 1, rho 1, 10 iterations) | 26.6 | 47.9 | 3.82 | 98.5% |
| spinning_3m | clarabel (same MPC as a sparse QP, interior point) | 1733.1 | 3859.9 | 3.82 | 98.5% |
| spinning_3m | osqp (same MPC as a sparse QP, ADMM, warm-started, eps 1e-4) | 973.6 | 15002.1 | 3.97 | 98.5% |
