/**
 * TypeScript twin of core/sirlab/solvers/*.py (Euler, Heun, RK4, and
 * adaptive RK45/Dormand-Prince). Stage combination formulas are written
 * literally as in the Python originals -- see parity.test.ts.
 */
import type { Model, Vector } from "./models";

export interface SolveResult {
  t: number[];
  y: Vector[];
  nFev: number;
  nSteps: number;
  diverged: boolean;
  negativityEvents: number;
}

function addScaled(a: Vector, scale: number, b: Vector): Vector {
  return a.map((v, i) => v + scale * b[i]);
}

function isFiniteVec(v: Vector): boolean {
  return v.every((x) => Number.isFinite(x));
}

function hasNegative(v: Vector): boolean {
  return v.some((x) => x < 0);
}

function uniformGrid(t0: number, tf: number, h: number): number[] {
  const n = Math.round((tf - t0) / h);
  return Array.from({ length: n + 1 }, (_, k) => t0 + h * k);
}

export function integrateEuler(model: Model, y0: Vector, theta: Vector, t0: number, tf: number, h: number): SolveResult {
  const t = uniformGrid(t0, tf, h);
  const y: Vector[] = [y0];
  let nFev = 0;
  let negativityEvents = 0;
  let diverged = false;
  for (let k = 0; k < t.length - 1; k++) {
    const yk = y[k];
    const f = model.rhs(t[k], yk, theta);
    nFev += 1;
    const yNext = addScaled(yk, h, f);
    if (hasNegative(yNext)) negativityEvents += 1;
    if (!isFiniteVec(yNext)) {
      diverged = true;
      y.push(yNext);
      break;
    }
    y.push(yNext);
  }
  return { t: t.slice(0, y.length), y, nFev, nSteps: y.length - 1, diverged, negativityEvents };
}

export function integrateHeun(model: Model, y0: Vector, theta: Vector, t0: number, tf: number, h: number): SolveResult {
  const t = uniformGrid(t0, tf, h);
  const y: Vector[] = [y0];
  let nFev = 0;
  let negativityEvents = 0;
  let diverged = false;
  for (let k = 0; k < t.length - 1; k++) {
    const yk = y[k];
    const f0 = model.rhs(t[k], yk, theta);
    const yPred = addScaled(yk, h, f0);
    const f1 = model.rhs(t[k + 1], yPred, theta);
    nFev += 2;
    const yNext = yk.map((v, i) => v + (h / 2.0) * (f0[i] + f1[i]));
    if (hasNegative(yNext)) negativityEvents += 1;
    if (!isFiniteVec(yNext)) {
      diverged = true;
      y.push(yNext);
      break;
    }
    y.push(yNext);
  }
  return { t: t.slice(0, y.length), y, nFev, nSteps: y.length - 1, diverged, negativityEvents };
}

export function integrateRK4(model: Model, y0: Vector, theta: Vector, t0: number, tf: number, h: number): SolveResult {
  const t = uniformGrid(t0, tf, h);
  const y: Vector[] = [y0];
  let nFev = 0;
  let negativityEvents = 0;
  let diverged = false;
  for (let k = 0; k < t.length - 1; k++) {
    const yk = y[k];
    const tk = t[k];
    const k1 = model.rhs(tk, yk, theta);
    const k2 = model.rhs(tk + h / 2.0, addScaled(yk, h / 2.0, k1), theta);
    const k3 = model.rhs(tk + h / 2.0, addScaled(yk, h / 2.0, k2), theta);
    const k4 = model.rhs(tk + h, addScaled(yk, h, k3), theta);
    nFev += 4;
    const yNext = yk.map((v, i) => v + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
    if (hasNegative(yNext)) negativityEvents += 1;
    if (!isFiniteVec(yNext)) {
      diverged = true;
      y.push(yNext);
      break;
    }
    y.push(yNext);
  }
  return { t: t.slice(0, y.length), y, nFev, nSteps: y.length - 1, diverged, negativityEvents };
}

// Dormand-Prince RK45 Butcher tableau, matching core/sirlab/solvers/rk45.py
const C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1];
const A: number[][] = [
  [],
  [1 / 5],
  [3 / 40, 9 / 40],
  [44 / 45, -56 / 15, 32 / 9],
  [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
  [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
];
const B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0];
const B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40];

export function integrateRK45(
  model: Model,
  y0: Vector,
  theta: Vector,
  t0: number,
  tf: number,
  opts: { rtol?: number; atol?: number; maxSteps?: number } = {}
): SolveResult {
  const rtol = opts.rtol ?? 1e-8;
  const atol = opts.atol ?? 1e-10;
  const maxSteps = opts.maxSteps ?? 200000;
  const n = y0.length;
  let y = y0.slice();
  let t = t0;
  const direction = tf >= t0 ? 1 : -1;

  const f0 = model.rhs(t0, y, theta);
  let h = direction * Math.min(0.01, Math.abs(tf - t0));
  let fPrev = f0;
  let nFev = 1;

  const ts = [t0];
  const ys: Vector[] = [y.slice()];
  let nSteps = 0;
  const safety = 0.9;
  const minFactor = 0.2;
  const maxFactor = 5.0;
  const order = 5;

  let guard = 0;
  while ((direction > 0 && t < tf - 1e-14) || (direction < 0 && t > tf + 1e-14)) {
    if ((direction > 0 && t + h > tf) || (direction < 0 && t + h < tf)) h = tf - t;
    guard += 1;
    if (guard > maxSteps) break;

    const k: Vector[] = [fPrev];
    for (let i = 1; i < 7; i++) {
      let yi = y.slice();
      for (let j = 0; j < A[i].length; j++) {
        yi = addScaled(yi, h * A[i][j], k[j]);
      }
      k.push(model.rhs(t + C[i] * h, yi, theta));
    }
    nFev += 6;

    const y5 = y.map((v, i) => v + h * B5.reduce((s, b, j) => s + b * k[j][i], 0));
    const y4 = y.map((v, i) => v + h * B4.reduce((s, b, j) => s + b * k[j][i], 0));
    const scale = y.map((v, i) => atol + rtol * Math.max(Math.abs(v), Math.abs(y5[i])));
    const errNorm = Math.sqrt(y5.reduce((s, _, i) => s + ((y5[i] - y4[i]) / scale[i]) ** 2, 0) / n);

    let factor: number;
    if (errNorm <= 1.0 || Math.abs(h) < 1e-14) {
      t += h;
      y = y5;
      fPrev = k[6];
      ts.push(t);
      ys.push(y.slice());
      nSteps += 1;
      factor = errNorm === 0 ? maxFactor : Math.min(maxFactor, Math.max(minFactor, safety * errNorm ** (-1 / (order + 1))));
    } else {
      factor = Math.max(minFactor, safety * errNorm ** (-1 / (order + 1)));
    }
    h *= factor;
  }

  const diverged = !isFiniteVec(y);
  return { t: ts, y: ys, nFev, nSteps, diverged, negativityEvents: 0 };
}

export type SolverName = "euler" | "heun" | "rk4" | "rk45";

export function integrate(
  model: Model,
  y0: Vector,
  theta: Vector,
  t0: number,
  tf: number,
  solver: SolverName,
  h?: number,
  rk45Opts?: { rtol?: number; atol?: number }
): SolveResult {
  switch (solver) {
    case "euler":
      return integrateEuler(model, y0, theta, t0, tf, h!);
    case "heun":
      return integrateHeun(model, y0, theta, t0, tf, h!);
    case "rk4":
      return integrateRK4(model, y0, theta, t0, tf, h!);
    case "rk45":
      return integrateRK45(model, y0, theta, t0, tf, rk45Opts);
  }
}

export const SOLVER_ORDER: Record<SolverName, number> = { euler: 1, heun: 2, rk4: 4, rk45: 5 };
