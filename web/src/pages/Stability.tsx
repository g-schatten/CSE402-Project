import { useMemo, useState } from "react";
import { PageHeader, Card, SectionTitle, Slider, Stat, Legend } from "../components/ui";
import { SIR } from "../numerics/models";
import { integrate, integrateRK4, type SolverName } from "../numerics/solvers";
import { LineChart } from "../charts/LineChart";

const N = 1000;
const BETA = 0.3;
const GAMMA = 0.1;

export default function Stability() {
  const [h, setH] = useState(10.0);
  const model = useMemo(() => new SIR(N), []);
  const y0 = useMemo(() => [N - 1, 1, 0], []);

  const eulerRes = integrate(model, y0, [BETA, GAMMA], 0, 100, "euler", h);
  const rk4Res = integrateRK4(model, y0, [BETA, GAMMA], 0, 100, Math.min(h, 0.5));

  const firstNegIdx = eulerRes.y.findIndex((row) => row.some((v) => v < 0));
  const firstFailureT = firstNegIdx >= 0 ? eulerRes.t[firstNegIdx] : null;

  const seriesEuler = ["S", "I", "R"].map((name, idx) => ({
    id: `Euler ${name}`,
    color: ["var(--c-S)", "var(--c-I)", "var(--c-R)"][idx],
    points: eulerRes.t.map((t, k) => ({ x: t, y: eulerRes.y[k][idx] })),
  }));

  // h * lambda_max along the RK4 (well-resolved) trajectory, for the stability-region view
  const hLambdaPoints = rk4Res.t.map((t, k) => {
    const [S, I] = rk4Res.y[k];
    // Jacobian eigenvalues for SIR at this point (2x2 relevant block, since R row is decoupled)
    const j11 = (-BETA * I) / N;
    const j12 = (-BETA * S) / N;
    const j21 = (BETA * I) / N;
    const j22 = (BETA * S) / N - GAMMA;
    const tr = j11 + j22;
    const det = j11 * j22 - j12 * j21;
    const disc = tr * tr - 4 * det;
    const lambdaMag = disc >= 0 ? Math.max(Math.abs((tr + Math.sqrt(disc)) / 2), Math.abs((tr - Math.sqrt(disc)) / 2)) : Math.sqrt(det);
    return { x: t, y: h * lambdaMag };
  });

  return (
    <div>
      <PageHeader eyebrow="03 · Interactive" title="Stability Playground">
        Crank up the step size and watch Forward Euler overshoot, go negative, then diverge — while RK4 at a modest
        step size stays well-behaved. The right panel tracks h·|λ| for the local Jacobian eigenvalues along the
        trajectory, linking the empirical failure to the textbook linear-stability picture.
      </PageHeader>

      <Card>
        <div className="control-row">
          <Slider label="Euler step size h" value={h} min={0.5} max={20} step={0.25} onChange={setH} />
        </div>
      </Card>

      <div className="grid-2">
        <Card>
          <SectionTitle sub={firstFailureT !== null ? `first negative compartment at t ≈ ${firstFailureT.toFixed(1)}` : "no negativity in this window"}>
            Forward Euler at h = {h.toFixed(2)}
          </SectionTitle>
          <LineChart width={520} height={300} xLabel="t (days)" yLabel="people" series={seriesEuler} formatX={(v) => v.toFixed(0)} formatY={(v) => v.toFixed(0)} />
          <Legend items={[{ label: "S", color: "var(--c-S)" }, { label: "I", color: "var(--c-I)" }, { label: "R", color: "var(--c-R)" }]} />
        </Card>
        <Card>
          <SectionTitle sub="Euler is stable (for this linearization) while h·|λ| < 2; forced by the chosen h regardless of t">
            h · |λ_max(t)| along the trajectory
          </SectionTitle>
          <LineChart
            width={520}
            height={300}
            xLabel="t (days)"
            yLabel="h·|λ|"
            series={[{ id: "h·|λ|", color: "var(--accent)", points: hLambdaPoints }]}
            markers={[{ x: 0, y: 2, color: "var(--bad)", label: "Euler stability boundary" }]}
            formatX={(v) => v.toFixed(0)}
            formatY={(v) => v.toFixed(2)}
          />
          <p style={{ color: "var(--text-secondary)", fontSize: "var(--fs-xs)" }}>
            Forward Euler's stability region requires |1 + z| ≤ 1 for z = hλ (real λ &lt; 0: |h·λ| ≤ 2). Once the
            curve exceeds 2, Euler amplifies error every step.
          </p>
        </Card>
      </div>

      <Card>
        <div className="grid-3">
          <Stat label="Euler negativity events" value={String(eulerRes.negativityEvents)} color={eulerRes.negativityEvents > 0 ? "var(--bad)" : "var(--good)"} />
          <Stat label="Euler diverged?" value={eulerRes.diverged ? "yes" : "no"} color={eulerRes.diverged ? "var(--bad)" : "var(--good)"} />
          <Stat label="RK4 (h≤0.5) negativity events" value={String(rk4Res.negativityEvents)} color="var(--good)" />
        </div>
      </Card>
    </div>
  );
}
