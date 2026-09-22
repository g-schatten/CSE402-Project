import { useMemo, useState } from "react";
import { PageHeader, Card, SectionTitle, Slider, Legend, Stat } from "../components/ui";
import { SIR } from "../numerics/models";
import { integrate, type SolverName } from "../numerics/solvers";
import { SIRReference } from "../numerics/reference";
import { LineChart } from "../charts/LineChart";

const N = 1000;
const BETA = 0.3;
const GAMMA = 0.1;
const SOLVERS: { id: SolverName; label: string; color: string; order: number }[] = [
  { id: "euler", label: "Euler", color: "var(--solver-euler)", order: 1 },
  { id: "heun", label: "Heun", color: "var(--solver-heun)", order: 2 },
  { id: "rk4", label: "RK4", color: "var(--solver-rk4)", order: 4 },
];

export default function SolverArena() {
  const [h, setH] = useState(1.0);
  const [tFinal] = useState(40);
  const model = useMemo(() => new SIR(N), []);
  const y0 = useMemo(() => [N - 1, 1, 0], []);
  const ref = useMemo(() => new SIRReference({ beta: BETA, gamma: GAMMA, nTotal: N, s0: y0[0], i0: y0[1] }), [y0]);

  const solverRuns = SOLVERS.map((s) => ({
    ...s,
    result: integrate(model, y0, [BETA, GAMMA], 0, tFinal, s.id, h),
  }));

  const refPoints = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    for (let k = 0; k <= 200; k++) {
      const t = (k / 200) * tFinal;
      pts.push({ x: t, y: ref.trajectoryAt(t).i });
    }
    return pts;
  }, [ref, tFinal]);

  const errorSeries = solverRuns.map((s) => ({
    id: s.label,
    color: s.color,
    points: s.result.t.map((t, k) => ({ x: t, y: Math.max(Math.abs(s.result.y[k][1] - ref.trajectoryAt(t).i), 1e-8) })),
  }));

  // Convergence (log-log): sweep h at several values for each solver, error at t_final
  const hSweep = [4, 2, 1, 0.5, 0.25, 0.125, 0.0625];
  const convergenceSeries = SOLVERS.map((s) => ({
    id: s.label,
    color: s.color,
    points: hSweep.map((hh) => {
      const res = integrate(model, y0, [BETA, GAMMA], 0, tFinal, s.id, hh);
      const truth = ref.trajectoryAt(tFinal).i;
      const err = Math.max(Math.abs(res.y[res.y.length - 1][1] - truth), 1e-10);
      return { x: hh, y: err };
    }),
  }));

  function fitSlope(points: { x: number; y: number }[]): number {
    const xs = points.map((p) => Math.log(p.x));
    const ys = points.map((p) => Math.log(p.y));
    const xm = xs.reduce((a, b) => a + b, 0) / xs.length;
    const ym = ys.reduce((a, b) => a + b, 0) / ys.length;
    const num = xs.reduce((s, x, i) => s + (x - xm) * (ys[i] - ym), 0);
    const den = xs.reduce((s, x) => s + (x - xm) ** 2, 0);
    return den ? num / den : NaN;
  }

  return (
    <div>
      <PageHeader eyebrow="02 · Interactive" title="Solver Arena">
        Race Forward Euler, Heun, and RK4 against the exact semi-analytic gold standard at a chosen step size h.
        Watch the error and the fitted convergence-order slope update live.
      </PageHeader>

      <Card>
        <div className="control-row">
          <Slider label="step size h" value={h} min={0.03} max={4} step={0.01} onChange={setH} />
        </div>
      </Card>

      <div className="grid-2">
        <Card>
          <SectionTitle sub="I(t): each solver vs the exact gold standard (dashed)">Trajectories</SectionTitle>
          <LineChart
            width={520}
            height={300}
            xLabel="t (days)"
            yLabel="I(t)"
            series={[
              { id: "exact", color: "var(--text-tertiary)", points: refPoints, dashed: true, width: 1.5 },
              ...solverRuns.map((s) => ({ id: s.label, color: s.color, points: s.result.t.map((t, k) => ({ x: t, y: s.result.y[k][1] })) })),
            ]}
            formatX={(v) => v.toFixed(0)}
            formatY={(v) => v.toFixed(0)}
          />
          <Legend items={[{ label: "exact", color: "var(--text-tertiary)" }, ...SOLVERS.map((s) => ({ label: s.label, color: s.color }))]} />
        </Card>
        <Card>
          <SectionTitle sub="log scale: |I_solver(t) − I_exact(t)|">Error over time</SectionTitle>
          <LineChart width={520} height={300} xLabel="t (days)" yLabel="|error|" yLog series={errorSeries} formatX={(v) => v.toFixed(0)} formatY={(v) => v.toExponential(1)} />
        </Card>
      </div>

      <Card>
        <SectionTitle sub="fitted slope ≈ observed convergence order p̂">Convergence (error at t=40 vs step size h, log-log)</SectionTitle>
        <LineChart width={1080} height={280} xLabel="h" yLabel="error at t=T" xLog yLog series={convergenceSeries} formatX={(v) => v.toString()} formatY={(v) => v.toExponential(1)} />
        <div className="grid-3" style={{ marginTop: 12 }}>
          {SOLVERS.map((s) => (
            <Stat key={s.id} label={`${s.label} observed order p̂`} value={fitSlope(convergenceSeries.find((c) => c.id === s.label)!.points).toFixed(2)} color={s.color} />
          ))}
        </div>
      </Card>

      <Card>
        <p style={{ color: "var(--text-secondary)", fontSize: "var(--fs-sm)", margin: 0 }}>
          Toggle the x-axis mentally between step size and cost: Euler needs 1 evaluation/step, Heun 2, RK4 4 — so
          RK4's steeper convergence wins on cost-adjusted accuracy too, not just per-step accuracy. See the{" "}
          <a href="#/explorer" style={{ color: "var(--accent-strong)" }}>
            Experiment Explorer
          </a>{" "}
          for the full factorial (E01) with work-precision diagrams.
        </p>
      </Card>
    </div>
  );
}
