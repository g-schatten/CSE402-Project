/**
 * TypeScript twin of core/sirlab/models/*.py. Every rhs() is written with
 * EXACTLY the same arithmetic ordering as the Python original so the two
 * implementations parity-test to ~1e-12 (see tests/parity.test.ts and
 * PLAN.md section 6.4). If you change one side's arithmetic, change both.
 */

export type Vector = number[];

export interface Model {
  name: string;
  stateNames: string[];
  paramNames: string[];
  nTotal: number;
  rhs(t: number, y: Vector, theta: Vector): Vector;
  r0(theta: Vector): number;
  initialState(opts: { i0: number; e0?: number; r0?: number; v0?: number }): Vector;
}

/** dS/dt = -beta S I / N ; dI/dt = beta S I / N - gamma I ; dR/dt = gamma I */
export class SIR implements Model {
  name = "SIR";
  stateNames = ["S", "I", "R"];
  paramNames = ["beta", "gamma"];
  constructor(public nTotal: number) {}

  rhs(_t: number, y: Vector, theta: Vector): Vector {
    const [S, I, _R] = y;
    const [beta, gamma] = theta;
    const N = this.nTotal;
    const infection = (beta * S * I) / N;
    const recovery = gamma * I;
    return [-infection, infection - recovery, recovery];
  }

  r0(theta: Vector): number {
    return theta[0] / theta[1];
  }

  initialState(opts: { i0: number }): Vector {
    const s0 = this.nTotal - opts.i0;
    return [s0, opts.i0, 0.0];
  }
}

/** SEIR: SIR plus a latent (exposed, non-infectious) compartment. */
export class SEIR implements Model {
  name = "SEIR";
  stateNames = ["S", "E", "I", "R"];
  paramNames = ["beta", "sigma_inc", "gamma"];
  constructor(public nTotal: number) {}

  rhs(_t: number, y: Vector, theta: Vector): Vector {
    const [S, E, I, _R] = y;
    const [beta, sigmaInc, gamma] = theta;
    const N = this.nTotal;
    const infection = (beta * S * I) / N;
    const progression = sigmaInc * E;
    const recovery = gamma * I;
    return [-infection, infection - progression, progression - recovery, recovery];
  }

  r0(theta: Vector): number {
    return theta[0] / theta[2];
  }

  initialState(opts: { i0: number; e0?: number }): Vector {
    const e0 = opts.e0 ?? 0;
    const s0 = this.nTotal - e0 - opts.i0;
    return [s0, e0, opts.i0, 0.0];
  }
}

/** SIRS: SIR plus waning immunity (xi * R feeding back into S). */
export class SIRS implements Model {
  name = "SIRS";
  stateNames = ["S", "I", "R"];
  paramNames = ["beta", "gamma", "xi"];
  constructor(public nTotal: number) {}

  rhs(_t: number, y: Vector, theta: Vector): Vector {
    const [S, I, R] = y;
    const [beta, gamma, xi] = theta;
    const N = this.nTotal;
    const infection = (beta * S * I) / N;
    const recovery = gamma * I;
    const waning = xi * R;
    return [-infection + waning, infection - recovery, recovery - waning];
  }

  r0(theta: Vector): number {
    return theta[0] / theta[1];
  }

  initialState(opts: { i0: number }): Vector {
    const s0 = this.nTotal - opts.i0;
    return [s0, opts.i0, 0.0];
  }
}

/** SIR + vaccination: S drains into an absorbing V class at rate nu. */
export class SIRVaccination implements Model {
  name = "SIR-V";
  stateNames = ["S", "I", "R", "V"];
  paramNames = ["beta", "gamma", "nu"];
  constructor(public nTotal: number) {}

  rhs(_t: number, y: Vector, theta: Vector): Vector {
    const [S, I, R, _V] = y;
    const [beta, gamma, nu] = theta;
    const N = this.nTotal;
    const infection = (beta * S * I) / N;
    const recovery = gamma * I;
    const vaccination = nu * S;
    return [-infection - vaccination, infection - recovery, recovery, vaccination];
  }

  r0(theta: Vector): number {
    return theta[0] / theta[1];
  }

  initialState(opts: { i0: number; v0?: number }): Vector {
    const v0 = opts.v0 ?? 0;
    const s0 = this.nTotal - opts.i0 - v0;
    return [s0, opts.i0, 0.0, v0];
  }
}

export const MODEL_REGISTRY = { SIR, SEIR, SIRS, "SIR-V": SIRVaccination } as const;
