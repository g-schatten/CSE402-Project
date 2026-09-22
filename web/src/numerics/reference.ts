/**
 * TypeScript twin of core/sirlab/reference.py: the semi-analytic SIR gold
 * standard (PLAN.md eq. 2.1-2.4). Simpler than the Python version (no
 * cached table -- the dashboard only ever needs a handful of query points
 * per interaction, not thousands of experiment-sweep calls), but the
 * underlying math is identical: adaptive Simpson quadrature for t(R),
 * safeguarded Newton (rtsafe) to invert it.
 */

export interface SIRReferenceParams {
  beta: number;
  gamma: number;
  nTotal: number;
  s0: number;
  i0: number;
}

/**
 * Iterative (worklist-based) adaptive Simpson's rule -- mirrors the fix
 * applied to core/sirlab/quadrature/adaptive.py after it was found to
 * silently under-converge (and, in an earlier recursive form, to risk
 * exponential blow-up) whenever tol-halving drove the per-panel budget
 * below floating-point roundoff before max_depth was reached. Both
 * safeguards from the Python fix are ported here: a hard cap on total
 * panel evaluations, and a roundoff floor on the acceptance criterion so a
 * panel is never forced to "converge" against a tolerance tighter than
 * double precision can express.
 */
function adaptiveSimpson(f: (x: number) => number, a: number, b: number, tol = 1e-12, maxDepth = 30, maxEvals = 20000): number {
  function simpson(a: number, b: number, fa: number, fm: number, fb: number): number {
    return ((b - a) / 6.0) * (fa + 4.0 * fm + fb);
  }
  if (a === b) return 0;
  const fa0 = f(a);
  const fb0 = f(b);
  const m0 = 0.5 * (a + b);
  const fm0 = f(m0);
  const whole0 = simpson(a, b, fa0, fm0, fb0);

  let total = 0;
  let evals = 3;
  type Panel = [number, number, number, number, number, number, number, number]; // a,b,fa,fm,fb,whole,tol,depth
  const stack: Panel[] = [[a, b, fa0, fm0, fb0, whole0, tol, maxDepth]];
  while (stack.length) {
    const [a_, b_, fa_, fm_, fb_, whole, tol_, depth] = stack.pop()!;
    const m = 0.5 * (a_ + b_);
    const lm = 0.5 * (a_ + m);
    const rm = 0.5 * (m + b_);
    const flm = f(lm);
    const frm = f(rm);
    evals += 2;
    const left = simpson(a_, m, fa_, flm, fm_);
    const right = simpson(m, b_, fm_, frm, fb_);
    const refined = left + right;
    const errEst = Math.abs(refined - whole);
    const roundoffFloor = 1e-14 * Math.max(Math.abs(refined), Math.abs(whole), 1.0);
    if (depth <= 0 || evals >= maxEvals || errEst <= Math.max(15.0 * tol_, roundoffFloor)) {
      total += refined + (refined - whole) / 15.0;
      continue;
    }
    stack.push([a_, m, fa_, flm, fm_, left, tol_ / 2.0, depth - 1]);
    stack.push([m, b_, fm_, frm, fb_, right, tol_ / 2.0, depth - 1]);
  }
  return total;
}

/** Safeguarded Newton (rtsafe), matching core/sirlab/roots/newton.py::newton_safe. */
function newtonSafe(f: (x: number) => number, fprime: (x: number) => number, lo0: number, hi0: number, tol = 1e-11, maxIter = 100): number {
  let lo = lo0;
  let hi = hi0;
  let flo = f(lo);
  const fhi = f(hi);
  if (flo === 0) return lo;
  if (fhi === 0) return hi;
  if (flo > 0) {
    [lo, hi] = [hi, lo];
    flo = f(lo);
  }
  let x = 0.5 * (lo + hi);
  let dxOld = Math.abs(hi - lo);
  let dx = dxOld;
  let fx = f(x);
  let dfx = fprime(x);
  for (let i = 0; i < maxIter; i++) {
    const outOfBracket = (x - hi) * dfx - fx !== 0 && ((x - hi) * dfx - fx) * ((x - lo) * dfx - fx) > 0;
    const tooSlow = Math.abs(2.0 * fx) > Math.abs(dxOld * dfx);
    let xNew: number;
    if (dfx === 0 || outOfBracket || tooSlow) {
      dxOld = dx;
      dx = 0.5 * (hi - lo);
      xNew = lo + dx;
      if (xNew === lo) return xNew;
    } else {
      dxOld = dx;
      dx = fx / dfx;
      xNew = x - dx;
      if (xNew === x) return xNew;
    }
    if (Math.abs(dx) < tol) return xNew;
    x = xNew;
    fx = f(x);
    dfx = fprime(x);
    if (fx < 0) lo = x;
    else hi = x;
  }
  return x;
}

export class SIRReference {
  constructor(private p: SIRReferenceParams) {}

  get r0Number(): number {
    return this.p.beta / this.p.gamma;
  }

  sOfR(r: number): number {
    const k = this.p.beta / (this.p.gamma * this.p.nTotal);
    return this.p.s0 * Math.exp(-k * r);
  }

  phaseInvariant(s: number, r: number): number {
    const k = this.p.beta / (this.p.gamma * this.p.nTotal);
    return Math.log(s) + k * r;
  }

  private dtDr(r: number): number {
    const s = this.sOfR(r);
    const i = Math.max(this.p.nTotal - r - s, 1e-300);
    return 1.0 / (this.p.gamma * i);
  }

  tOfR(r: number): number {
    if (r <= 0) return 0;
    return adaptiveSimpson((x) => this.dtDr(x), 0, r);
  }

  get sInfinity(): number {
    const r0n = this.r0Number;
    const n = this.p.nTotal;
    const g = (s: number) => Math.log(this.p.s0 / s) - (r0n * (n - s)) / n;
    const gp = (s: number) => -1.0 / s + r0n / n;
    return newtonSafe(g, gp, 1e-9 * n, this.p.s0);
  }

  get rInfinity(): number {
    return this.p.nTotal - this.sInfinity;
  }

  rOfT(t: number): number {
    if (t <= 0) return 0;
    const rInf = this.rInfinity;
    let frac = 0.5;
    let hi = rInf * frac * (1 - 1e-9);
    let tHi = this.tOfR(hi);
    let tries = 0;
    while (tHi < t && frac < 1 - 1e-14 && tries < 60) {
      frac = 1 - (1 - frac) / 2;
      hi = rInf * frac * (1 - 1e-9);
      tHi = this.tOfR(hi);
      tries += 1;
    }
    if (tHi < t) return hi;
    const f = (r: number) => this.tOfR(r) - t;
    const fp = (r: number) => this.dtDr(r);
    return newtonSafe(f, fp, 0, hi, 1e-10 * Math.max(1, t));
  }

  trajectoryAt(t: number): { s: number; i: number; r: number } {
    const r = this.rOfT(t);
    const s = this.sOfR(r);
    const i = this.p.nTotal - s - r;
    return { s, i, r };
  }

  get peakI(): number {
    const r0n = this.r0Number;
    const { nTotal: n, s0, i0 } = this.p;
    const sAtPeak = n / r0n;
    return i0 + s0 - sAtPeak + sAtPeak * Math.log(n / (r0n * s0));
  }

  peakTime(): number {
    const r0n = this.r0Number;
    const sTarget = this.p.nTotal / r0n;
    const k = this.p.beta / (this.p.gamma * this.p.nTotal);
    const rAtPeak = -Math.log(sTarget / this.p.s0) / k;
    return this.tOfR(rAtPeak);
  }
}
