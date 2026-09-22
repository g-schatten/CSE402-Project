import { useMemo, useState } from "react";
import { PageHeader, Card, SectionTitle } from "../components/ui";
import { useExperiment } from "../state/useExperiment";
import { LineChart } from "../charts/LineChart";

interface Cell {
  solver: string;
  h: number;
  sigma_frac: number;
  dt: number;
  beta_true: number;
  gamma_true: number;
  n_converged: number;
  n_total_trials: number;
  bias: (number | null)[];
  variance: (number | null)[];
  rmse: (number | null)[];
}
interface E07Payload {
  n_cells: number;
  n_replicates: number;
  cells: Cell[];
}

const SOLVER_COLOR: Record<string, string> = { euler: "var(--solver-euler)", heun: "var(--solver-heun)", rk4: "var(--solver-rk4)" };

export default function ExperimentExplorer() {
  const { data, loading, error } = useExperiment<E07Payload>("E07");
  const [solverFilter, setSolverFilter] = useState<string | null>(null);

  const solvers = useMemo(() => Array.from(new Set(data?.cells.map((c) => c.solver) ?? [])), [data]);
  const thetaKeys = useMemo(
    () => Array.from(new Set(data?.cells.map((c) => `${c.beta_true},${c.gamma_true}`) ?? [])),
    [data]
  );
  const [thetaFilter, setThetaFilter] = useState<string | null>(null);

  const filtered = (data?.cells ?? []).filter(
    (c) => (!solverFilter || c.solver === solverFilter) && (!thetaFilter || `${c.beta_true},${c.gamma_true}` === thetaFilter)
  );

  // RMSE(beta) vs sigma_frac, one series per solver, at fixed h (the smallest available) and dt
  const hValues = Array.from(new Set(filtered.map((c) => c.h))).sort((a, b) => a - b);
  const smallestH = hValues[0];
  const seriesRmseBeta = solvers
    .filter((s) => !solverFilter || s === solverFilter)
    .map((s) => ({
      id: s,
      color: SOLVER_COLOR[s] ?? "var(--accent)",
      points: filtered
        .filter((c) => c.solver === s && c.h === smallestH)
        .sort((a, b) => a.sigma_frac - b.sigma_frac)
        .map((c) => ({ x: c.sigma_frac, y: Math.max(c.rmse[0] ?? 1e-6, 1e-6) })),
    }));

  return (
    <div>
      <PageHeader eyebrow="06 · Precomputed (E07)" title="Experiment Explorer">
        The full solver-in-the-loop recovery factorial, exposed: click a solver or a (β, γ) regime to filter every
        panel. This is where a reader can convince themselves the sweep was actually run.
      </PageHeader>

      {loading && <Card>Loading…</Card>}
      {error && <Card>Could not load E07 ({error}).</Card>}

      {data && (
        <>
          <Card>
            <SectionTitle>{data.n_cells} cells × {data.n_replicates} replicates each</SectionTitle>
            <div className="pill-row">
              <button className={"btn" + (!solverFilter ? " active" : "")} onClick={() => setSolverFilter(null)}>
                all solvers
              </button>
              {solvers.map((s) => (
                <button key={s} className={"btn" + (solverFilter === s ? " active" : "")} onClick={() => setSolverFilter(s)}>
                  {s}
                </button>
              ))}
            </div>
            <div className="pill-row" style={{ marginTop: 8 }}>
              <button className={"btn" + (!thetaFilter ? " active" : "")} onClick={() => setThetaFilter(null)}>
                all (β,γ)
              </button>
              {thetaKeys.map((k) => (
                <button key={k} className={"btn" + (thetaFilter === k ? " active" : "")} onClick={() => setThetaFilter(k)}>
                  β,γ = {k}
                </button>
              ))}
            </div>
          </Card>

          <Card>
            <SectionTitle sub={`h = ${smallestH} (smallest tested), dt & θ combined across the filter`}>RMSE(β̂) vs noise level σ</SectionTitle>
            <LineChart width={1080} height={320} xLabel="σ (fraction of max I)" yLabel="RMSE(β̂)" yLog series={seriesRmseBeta} formatX={(v) => v.toFixed(3)} formatY={(v) => v.toExponential(1)} />
          </Card>

          <Card>
            <SectionTitle sub="each row = one (solver, h, σ, dt, β, γ) cell">Raw cells (first 40, filtered)</SectionTitle>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "var(--fs-xs)" }} className="mono">
                <thead>
                  <tr style={{ textAlign: "left", color: "var(--text-secondary)" }}>
                    {["solver", "h", "σ", "dt", "β", "γ", "bias(β)", "bias(γ)", "n_ok"].map((h) => (
                      <th key={h} style={{ padding: "4px 8px", borderBottom: "1px solid var(--border)" }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.slice(0, 40).map((c, i) => (
                    <tr key={i}>
                      <td style={{ padding: "4px 8px", color: SOLVER_COLOR[c.solver] }}>{c.solver}</td>
                      <td style={{ padding: "4px 8px" }}>{c.h}</td>
                      <td style={{ padding: "4px 8px" }}>{c.sigma_frac}</td>
                      <td style={{ padding: "4px 8px" }}>{c.dt}</td>
                      <td style={{ padding: "4px 8px" }}>{c.beta_true}</td>
                      <td style={{ padding: "4px 8px" }}>{c.gamma_true}</td>
                      <td style={{ padding: "4px 8px" }}>{c.bias[0]?.toExponential(2) ?? "—"}</td>
                      <td style={{ padding: "4px 8px" }}>{c.bias[1]?.toExponential(2) ?? "—"}</td>
                      <td style={{ padding: "4px 8px" }}>
                        {c.n_converged}/{c.n_total_trials}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
