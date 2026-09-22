import { useMemo, useState } from "react";
import { PageHeader, Card, SectionTitle, Stat } from "../components/ui";
import { useExperiment } from "../state/useExperiment";
import { LineChart } from "../charts/LineChart";

interface Row {
  h: number;
  solver_bias: number[];
  sigma_curve: { sigma_frac: number; std: number[]; mean_deviation: number[]; n_converged: number }[];
  crossover_sigma_star: { beta: number | null; gamma: number | null };
}
interface E13Payload {
  solver: string;
  theta_true: number[];
  conclusion: string;
  rows: Row[];
}

export default function Crossover() {
  const { data, loading, error } = useExperiment<E13Payload>("E13");
  const [hIdx, setHIdx] = useState(0);

  const row = data?.rows[Math.min(hIdx, data.rows.length - 1)];

  const stdLine = row
    ? row.sigma_curve.filter((c) => c.sigma_frac > 0).map((c) => ({ x: c.sigma_frac, y: c.std[0] }))
    : [];
  // Span the dashed bias reference line across exactly the plotted sigma
  // range (not an arbitrary wide domain) so it doesn't force the x-axis to
  // stretch far beyond the actual data and crowd the tick labels.
  const sigmaExtent = stdLine.length ? [stdLine[0].x, stdLine[stdLine.length - 1].x] : [1e-3, 1e-1];
  const biasLine = row
    ? [
        { x: sigmaExtent[0], y: Math.abs(row.solver_bias[0]) },
        { x: sigmaExtent[1], y: Math.abs(row.solver_bias[0]) },
      ]
    : [];

  return (
    <div>
      <PageHeader eyebrow="07 · Precomputed (E13) · The headline finding" title="Crossover σ*">
        Solver-induced bias (flat line, set by step size h alone) versus noise-induced standard deviation (rising
        curve, set by σ) on the same axes. Their intersection is σ*(h): below it, solver choice dominates the total
        parameter error; above it, sampling noise swamps any solver-accuracy difference.
      </PageHeader>

      {loading && <Card>Loading…</Card>}
      {error && <Card>Could not load E13 ({error}).</Card>}

      {data && row && (
        <>
          <Card>
            <div className="control-row">
              <div className="control">
                <label>
                  <span>fitting step size h ({data.solver})</span>
                  <span className="value">{row.h}</span>
                </label>
                <input type="range" min={0} max={data.rows.length - 1} step={1} value={hIdx} onChange={(e) => setHIdx(parseInt(e.target.value))} />
              </div>
            </div>
          </Card>

          <Card>
            <SectionTitle sub="both series for parameter β; log-log">Bias (solver truncation error) vs noise-induced std</SectionTitle>
            <LineChart
              width={1080}
              height={340}
              xLabel="σ (noise fraction)"
              yLabel="|error| in β̂"
              xLog
              yLog
              series={[
                { id: "solver bias(h) — deterministic", color: "var(--bad)", points: biasLine, dashed: true },
                { id: "std across replicates", color: "var(--accent)", points: stdLine },
              ]}
              markers={row.crossover_sigma_star.beta ? [{ x: row.crossover_sigma_star.beta, y: Math.abs(row.solver_bias[0]), color: "var(--good)", label: "σ*" }] : []}
              formatX={(v) => v.toExponential(1)}
              formatY={(v) => v.toExponential(1)}
            />
          </Card>

          <div className="grid-2">
            <Card>
              <Stat label="solver bias(h) in β" value={row.solver_bias[0].toExponential(3)} color="var(--bad)" />
            </Card>
            <Card>
              <Stat label="crossover σ*(h) for β" value={row.crossover_sigma_star.beta?.toFixed(4) ?? "not reached in this sweep"} color="var(--good)" />
            </Card>
          </div>

          <Card>
            <p style={{ color: "var(--text-secondary)", margin: 0 }}>{data.conclusion}</p>
          </Card>
        </>
      )}
    </div>
  );
}
