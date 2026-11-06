//! Our ballistics and gimbal-trajectory solvers against existing crates, on our own problems.
//!
//! `cargo run --release` from this directory. Needs `out/runs/{translate_spinning,spinning_3m}.jsonl`
//! from the main crate (the aim points the pipeline produced), used as the gimbal reference.

use std::hint::black_box;
use std::time::Instant;

use auto_aim::aimer::ballistics;
use auto_aim::aimer::trajectory::Axis;
use na34::{SMatrix, SVector};
use nalgebra::Vector3;
use ode_solvers::{Dopri5, Rk4, System, Vector4};

const G: f64 = 9.81;
const K: f64 = 0.019;
const V: f64 = 23.0;

// ---------------------------------------------------------------- ballistics

/// Flight with range as the independent variable, so integration ends exactly at the target.
/// State `[height, vx, vy, time]`.
struct Flight;

impl System<f64, Vector4<f64>> for Flight {
    fn system(&self, _x: f64, s: &Vector4<f64>, d: &mut Vector4<f64>) {
        let (vx, vy) = (s[1], s[2]);
        let v = vx.hypot(vy);
        d[0] = vy / vx;
        d[1] = -K * v;
        d[2] = (-K * v * vy - G) / vx;
        d[3] = 1.0 / vx;
    }
}

fn start(pitch: f64) -> Vector4<f64> {
    Vector4::new(0.0, V * pitch.cos(), V * pitch.sin(), 0.0)
}

fn dopri_height(range: f64, pitch: f64, tol: f64) -> f64 {
    let mut s = Dopri5::new(Flight, 0.0, range, range, start(pitch), tol, tol);
    s.integrate().map_or(f64::NAN, |_| s.y_out().last().map_or(f64::NAN, |y| y[0]))
}

fn rk4_height(range: f64, pitch: f64, max_step: f64) -> f64 {
    let n = (range / max_step).ceil();
    let mut s = Rk4::new(Flight, 0.0, start(pitch), range, range / n);
    s.integrate().map_or(f64::NAN, |_| s.y_out().last().map_or(f64::NAN, |y| y[0]))
}

fn vacuum_pitch(d: f64, h: f64) -> f64 {
    let a = G * d * d / (2.0 * V * V);
    ((d - (d * d - 4.0 * a * (a + h)).sqrt()) / (2.0 * a)).atan()
}

/// awakening's closed form: drag on the horizontal velocity only.
fn awakening_pitch(d: f64, h: f64) -> f64 {
    let a = (K * d).exp_m1() / K;
    let c = a / V;
    let b = 0.5 * G * c * c;
    ((a - (a * a - 4.0 * b * (b + h)).sqrt()) / (2.0 * b)).atan()
}

struct ArgminMiss {
    d: f64,
    h: f64,
}

impl argmin::core::CostFunction for ArgminMiss {
    type Param = f64;
    type Output = f64;
    fn cost(&self, pitch: &f64) -> Result<f64, argmin::core::Error> {
        Ok(ballistics::height_at(self.d, V, *pitch, K).map_or(f64::NAN, |(z, _)| z) - self.h)
    }
}

fn bench_ballistics() {
    let targets: Vec<(f64, f64)> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].iter().flat_map(|&d| [-0.4, 0.0, 0.3].map(|h| (d, h))).collect();
    let reference = |d: f64, h: f64| {
        roots::find_root_brent(-0.6, 0.7, |p| dopri_height(d, p, 1e-12) - h, &mut roots::SimpleConvergency { eps: 1e-14, max_iter: 200 }).expect("reference converges")
    };
    let refs: Vec<f64> = targets.iter().map(|&(d, h)| reference(d, h)).collect();

    type Method = (&'static str, Box<dyn Fn(f64, f64) -> f64>);
    let tol = || roots::SimpleConvergency { eps: 1e-7, max_iter: 60 };
    let methods: Vec<Method> = vec![
        ("ours: RK4 5 ms + secant from vacuum pitch", Box::new(|d, h| ballistics::solve(&Vector3::new(d, 0.0, h), V, K).map_or(f64::NAN, |s| s.pitch))),
        ("ode_solvers Dopri5 (tol 1e-8) + roots Brent", Box::new(move |d, h| roots::find_root_brent(-0.6, 0.7, |p| dopri_height(d, p, 1e-8) - h, &mut tol()).unwrap_or(f64::NAN))),
        ("ode_solvers Dopri5 (tol 1e-8) + roots secant from vacuum pitch", Box::new(move |d, h| {
            let p0 = vacuum_pitch(d, h);
            roots::find_root_secant(p0, p0 + 0.01, |p| dopri_height(d, p, 1e-8) - h, &mut tol()).unwrap_or(f64::NAN)
        })),
        ("ode_solvers RK4 (0.1 m steps) + roots secant from vacuum pitch", Box::new(move |d, h| {
            let p0 = vacuum_pitch(d, h);
            roots::find_root_secant(p0, p0 + 0.01, |p| rk4_height(d, p, 0.1) - h, &mut tol()).unwrap_or(f64::NAN)
        })),
        ("our integrator + argmin BrentRoot", Box::new(|d, h| {
            let solver = argmin::solver::brent::BrentRoot::new(-0.6, 0.7, 1e-9);
            use argmin::core::State as _;
            argmin::core::Executor::new(ArgminMiss { d, h }, solver)
                .configure(|s| s.param(vacuum_pitch(d, h)).max_iters(60))
                .run()
                .ok()
                .and_then(|r| r.state().get_best_param().copied())
                .unwrap_or(f64::NAN)
        })),
        ("awakening closed form (no simulation)", Box::new(awakening_pitch)),
    ];

    println!("## Ballistics: pitch to hit a target, 23 m/s, k = 0.019, ranges 2-8 m, heights -0.4/0/+0.3 m\n");
    println!("Miss = vertical miss at the target when the solved pitch is flown by a 1e-12 Dopri5 reference.\n");
    println!("| Method | µs per solve | worst miss mm | median miss mm |");
    println!("|---|---|---|---|");
    for (name, f) in &methods {
        let reps = 200;
        let start = Instant::now();
        for _ in 0..reps {
            for &(d, h) in &targets {
                black_box(f(black_box(d), black_box(h)));
            }
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(reps) / targets.len() as f64;
        let mut miss: Vec<f64> = targets
            .iter()
            .zip(&refs)
            .map(|(&(d, h), _)| (dopri_height(d, f(d, h), 1e-12) - h).abs() * 1e3)
            .collect();
        miss.sort_by(f64::total_cmp);
        println!("| {name} | {us:.1} | {:.3} | {:.3} |", miss[miss.len() - 1], miss[miss.len() / 2]);
    }
}

// ---------------------------------------------------------------- gimbal trajectory

const DT: f64 = 0.005;
const N: usize = 100;
const A_MAX: f64 = 60.0;
const V_MAX: f64 = 10.0;
/// sp_vision's weights (Q = diag(9e6, 0), R = 1), divided through by 9e6.
const R_WEIGHT: f64 = 1.0 / 9e6;

/// Aim points the pipeline produced, one per frame, held between frames.
struct Reference {
    t: Vec<f64>,
    yaw: Vec<f64>,
    tol: Vec<f64>,
}

impl Reference {
    fn load(video: &str) -> Reference {
        let path = format!("{}/../../out/runs/{video}.jsonl", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let (mut t, mut yaw, mut tol): (Vec<f64>, Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());
        for line in text.lines() {
            let row: serde_json::Value = serde_json::from_str(line).expect("json");
            let target = &row["record"]["target"];
            if target.is_null() {
                continue;
            }
            let mut y = target["shot"]["yaw"].as_f64().expect("yaw");
            if let Some(&prev) = yaw.last() {
                y = prev + (y - prev + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
            }
            let p: Vec<f64> = target["position"].as_array().expect("position").iter().map(|v| v.as_f64().expect("f64")).collect();
            let distance = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            t.push(row["t"].as_f64().expect("t"));
            yaw.push(y);
            tol.push(0.5 * 0.135 * target["facing"].as_f64().expect("facing").cos() / distance);
        }
        Reference { t, yaw, tol }
    }

    fn index(&self, tau: f64) -> usize {
        self.t.partition_point(|&t| t <= tau).saturating_sub(1)
    }

    fn window(&self, tau: f64) -> [f64; N + 1] {
        std::array::from_fn(|k| self.yaw[self.index(tau + k as f64 * DT)])
    }
}

trait Planner {
    fn name(&self) -> &'static str;
    /// Commanded acceleration for this tick from the gimbal state `[angle, rate]`.
    fn accel(&mut self, x: [f64; 2], r: &[f64; N + 1]) -> f64;
}

/// Ours: the critically damped, acceleration-clamped smoother, with the aimer's 10 ms velocity feedforward.
struct Smoother;

impl Planner for Smoother {
    fn name(&self) -> &'static str {
        "ours: reactive smoother (bandwidth 60)"
    }
    fn accel(&mut self, x: [f64; 2], r: &[f64; N + 1]) -> f64 {
        let mut axis = Axis { pos: x[0], vel: x[1], acc: 0.0 };
        axis.step(&Axis { pos: r[0], vel: (r[2] - r[0]) / 0.01, acc: 0.0 }, DT, A_MAX, 60.0);
        axis.acc
    }
}

type Cache = tinympc_rs::policy::ArrayPolicy<f64, 2, 1, 5>;
type Tiny = tinympc_rs::Solver<f64, Cache, 2, 1, N, { N - 1 }>;

struct TinyMpc {
    mpc: Tiny,
}

impl TinyMpc {
    fn new() -> TinyMpc {
        let a = SMatrix::<f64, 2, 2>::new(1.0, DT, 0.0, 1.0);
        let b = SMatrix::<f64, 2, 1>::new(0.5 * DT * DT, DT);
        let q = SMatrix::<f64, 2, 2>::new(9e6, 0.0, 0.0, 0.0);
        let r = SMatrix::<f64, 1, 1>::new(1.0);
        let cache = Cache::new(1.0, 10.0, 2.0, 1000, &a, &b, &q, &r, &SMatrix::zeros()).expect("tinympc setup");
        let mut mpc = Tiny::new(a, b, cache);
        mpc.config.max_iter = 10;
        TinyMpc { mpc }
    }
}

impl Planner for TinyMpc {
    fn name(&self) -> &'static str {
        "tinympc-rs (sp_vision's MPC: Q 9e6, R 1, rho 1, 10 iterations)"
    }
    fn accel(&mut self, x: [f64; 2], r: &[f64; N + 1]) -> f64 {
        use tinympc_rs::project::{Box as Bounds, ProjectMultiExt as _, dim::Lift, time::Fixed};
        let x_ref = SMatrix::<f64, 2, N>::from_fn(|i, k| if i == 0 { r[k] } else { 0.0 });
        let u_box = Lift::new([0], Bounds { upper: SVector::<f64, 1>::new(A_MAX), lower: SVector::<f64, 1>::new(-A_MAX) });
        let v_box = Lift::new([1], Bounds { upper: SVector::<f64, 1>::new(V_MAX), lower: SVector::<f64, 1>::new(-V_MAX) });
        let mut u_con = [Fixed::new(&u_box).constraint_owned()];
        let mut x_con = [Fixed::new(&v_box).constraint_owned()];
        let solution = self.mpc.initial_condition(SVector::<f64, 2>::new(x[0], x[1])).u_constraints(&mut u_con).x_constraints(&mut x_con).x_reference(&x_ref).solve();
        solution.u_now()[0]
    }
}

/// The same MPC as one sparse QP. Variables `[p_1..p_N, v_1..v_N, u_0..u_{N-1}]`.
struct Qp {
    p_upper: Vec<(usize, usize, f64)>,
    a: Vec<(usize, usize, f64)>,
    eq_rows: usize,
}

const fn ip(k: usize) -> usize {
    k - 1
}
const fn iv(k: usize) -> usize {
    N + k - 1
}
const fn iu(k: usize) -> usize {
    2 * N + k
}

impl Qp {
    fn new() -> Qp {
        let mut p_upper = Vec::new();
        for k in 1..=N {
            p_upper.push((ip(k), ip(k), 2.0));
        }
        for k in 0..N {
            p_upper.push((iu(k), iu(k), 2.0 * R_WEIGHT));
        }
        let mut a = Vec::new();
        for k in 0..N {
            a.push((2 * k, ip(k + 1), 1.0));
            a.push((2 * k, iu(k), -0.5 * DT * DT));
            a.push((2 * k + 1, iv(k + 1), 1.0));
            a.push((2 * k + 1, iu(k), -DT));
            if k > 0 {
                a.push((2 * k, ip(k), -1.0));
                a.push((2 * k, iv(k), -DT));
                a.push((2 * k + 1, iv(k), -1.0));
            }
        }
        Qp { p_upper, a, eq_rows: 2 * N }
    }

    fn q(r: &[f64; N + 1]) -> Vec<f64> {
        let mut q = vec![0.0; 3 * N];
        for k in 1..=N {
            q[ip(k)] = -2.0 * r[k];
        }
        q
    }

    fn b_eq(x: [f64; 2]) -> Vec<f64> {
        let mut b = vec![0.0; 2 * N];
        b[0] = x[0] + DT * x[1];
        b[1] = x[1];
        b
    }
}

fn csc(m: usize, n: usize, mut t: Vec<(usize, usize, f64)>) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    t.sort_by_key(|&(r, c, _)| (c, r));
    let mut colptr = vec![0; n + 1];
    for &(_, c, _) in &t {
        colptr[c + 1] += 1;
    }
    for c in 0..n {
        colptr[c + 1] += colptr[c];
    }
    assert!(t.iter().all(|&(r, _, _)| r < m), "row in range");
    (colptr, t.iter().map(|e| e.0).collect(), t.iter().map(|e| e.2).collect())
}

struct Clarabel {
    solver: clarabel::solver::DefaultSolver<f64>,
}

impl Clarabel {
    fn new() -> Clarabel {
        use clarabel::algebra::CscMatrix;
        use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, SupportedConeT};
        let qp = Qp::new();
        let (cp, rp, vp) = csc(3 * N, 3 * N, qp.p_upper.clone());
        let mut a = qp.a.clone();
        for k in 0..N {
            a.push((qp.eq_rows + 2 * k, iu(k), 1.0));
            a.push((qp.eq_rows + 2 * k + 1, iu(k), -1.0));
            a.push((qp.eq_rows + 2 * N + 2 * k, iv(k + 1), 1.0));
            a.push((qp.eq_rows + 2 * N + 2 * k + 1, iv(k + 1), -1.0));
        }
        let m = qp.eq_rows + 4 * N;
        let (ca, ra, va) = csc(m, 3 * N, a);
        let mut b = Qp::b_eq([0.0, 0.0]);
        b.extend((0..N).flat_map(|_| [A_MAX, A_MAX]));
        b.extend((0..N).flat_map(|_| [V_MAX, V_MAX]));
        let cones = [SupportedConeT::ZeroConeT(qp.eq_rows), SupportedConeT::NonnegativeConeT(4 * N)];
        let settings = DefaultSettingsBuilder::default().verbose(false).build().expect("settings");
        let solver = DefaultSolver::new(
            &CscMatrix::new(3 * N, 3 * N, cp, rp, vp),
            &vec![0.0; 3 * N],
            &CscMatrix::new(m, 3 * N, ca, ra, va),
            &b,
            &cones,
            settings,
        )
        .expect("clarabel setup");
        Clarabel { solver }
    }
}

impl Planner for Clarabel {
    fn name(&self) -> &'static str {
        "clarabel (same MPC as a sparse QP, interior point)"
    }
    fn accel(&mut self, x: [f64; 2], r: &[f64; N + 1]) -> f64 {
        use clarabel::solver::IPSolver;
        let mut b = Qp::b_eq(x);
        b.extend((0..N).flat_map(|_| [A_MAX, A_MAX]));
        b.extend((0..N).flat_map(|_| [V_MAX, V_MAX]));
        self.solver.update_q(&Qp::q(r)).expect("update q");
        self.solver.update_b(&b).expect("update b");
        self.solver.solve();
        self.solver.solution.x[iu(0)]
    }
}

struct Osqp {
    problem: osqp::Problem,
}

impl Osqp {
    fn bounds(x: [f64; 2]) -> (Vec<f64>, Vec<f64>) {
        let eq = Qp::b_eq(x);
        let mut l = eq.clone();
        let mut u = eq;
        l.extend(std::iter::repeat_n(-A_MAX, N).chain(std::iter::repeat_n(-V_MAX, N)));
        u.extend(std::iter::repeat_n(A_MAX, N).chain(std::iter::repeat_n(V_MAX, N)));
        (l, u)
    }

    fn new() -> Osqp {
        let qp = Qp::new();
        let (cp, rp, vp) = csc(3 * N, 3 * N, qp.p_upper.clone());
        let mut a = qp.a.clone();
        for k in 0..N {
            a.push((qp.eq_rows + k, iu(k), 1.0));
            a.push((qp.eq_rows + N + k, iv(k + 1), 1.0));
        }
        let m = qp.eq_rows + 2 * N;
        let (ca, ra, va) = csc(m, 3 * N, a);
        let p = osqp::CscMatrix { nrows: 3 * N, ncols: 3 * N, indptr: cp.into(), indices: rp.into(), data: vp.into() };
        let a = osqp::CscMatrix { nrows: m, ncols: 3 * N, indptr: ca.into(), indices: ra.into(), data: va.into() };
        let (l, u) = Osqp::bounds([0.0, 0.0]);
        let settings = osqp::Settings::default().verbose(false).polishing(false).eps_abs(1e-4).eps_rel(1e-4).max_iter(4000);
        Osqp { problem: osqp::Problem::new(p, &vec![0.0; 3 * N], a, &l, &u, &settings).expect("osqp setup") }
    }
}

impl Planner for Osqp {
    fn name(&self) -> &'static str {
        "osqp (same MPC as a sparse QP, ADMM, warm-started, eps 1e-4)"
    }
    fn accel(&mut self, x: [f64; 2], r: &[f64; N + 1]) -> f64 {
        let (l, u) = Osqp::bounds(x);
        self.problem.update_lin_cost(&Qp::q(r));
        self.problem.update_bounds(&l, &u);
        self.problem.solve().x().map_or(0.0, |z| z[iu(0)])
    }
}

fn bench_trajectory() {
    println!("\n## Gimbal yaw trajectory: closed loop at 200 Hz against the recorded aim points\n");
    println!("Gimbal = ideal double integrator, |acc| <= {A_MAX} rad/s², |rate| <= {V_MAX} rad/s. MPC horizon {N} x {} ms, and it is given the *recorded* future aim points (perfect foresight). In the pipeline that future would come from the robot model.\n", DT * 1e3);
    println!("| Video | Planner | µs per tick, median | p99 | RMS error mrad | ticks within fire tolerance |");
    println!("|---|---|---|---|---|---|");
    let mut csv = String::from("video,planner,t,reference,command\n");
    for video in ["translate_spinning", "spinning_3m"] {
        let reference = Reference::load(video);
        let planners: Vec<Box<dyn Planner>> = vec![Box::new(Smoother), Box::new(TinyMpc::new()), Box::new(Clarabel::new()), Box::new(Osqp::new())];
        for mut planner in planners {
            let (t0, t1) = (reference.t[0], reference.t[reference.t.len() - 1] - N as f64 * DT);
            let mut x = [reference.yaw[0], 0.0];
            let (mut times, mut sq, mut within, mut ticks) = (Vec::new(), 0.0, 0usize, 0usize);
            let mut tau = t0;
            while tau < t1 {
                let r = reference.window(tau);
                let start = Instant::now();
                let a = planner.accel(x, &r).clamp(-A_MAX, A_MAX);
                times.push(start.elapsed().as_secs_f64() * 1e6);
                x = [x[0] + x[1] * DT + 0.5 * a * DT * DT, (x[1] + a * DT).clamp(-V_MAX, V_MAX)];
                tau += DT;
                let (target, tol) = (reference.yaw[reference.index(tau)], reference.tol[reference.index(tau)]);
                let err = x[0] - target;
                sq += err * err;
                within += usize::from(err.abs() < tol);
                ticks += 1;
                if (6.0..8.0).contains(&tau) || video == "spinning_3m" && (4.0..6.0).contains(&tau) {
                    csv.push_str(&format!("{video},{},{tau:.4},{target},{}\n", planner.name(), x[0]));
                }
            }
            times.sort_by(f64::total_cmp);
            println!(
                "| {video} | {} | {:.1} | {:.1} | {:.2} | {:.1}% |",
                planner.name(),
                times[times.len() / 2],
                times[times.len() * 99 / 100],
                (sq / ticks as f64).sqrt() * 1e3,
                100.0 * within as f64 / ticks as f64
            );
        }
    }
    let out = format!("{}/../../out/solver_bench_trajectory.csv", env!("CARGO_MANIFEST_DIR"));
    std::fs::write(&out, csv).expect("write csv");
}

fn main() {
    bench_ballistics();
    bench_trajectory();
}
