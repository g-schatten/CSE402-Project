import { useState } from "react";
import { PageHeader, Card, SectionTitle, Stat } from "../components/ui";
import { useExperiment } from "../state/useExperiment";
import { LineChart } from "../charts/LineChart";

interface E06Payload {
  theta_true: number[];
  results: Record<string, { mean_iterations: number; mean_fevals: number; success_rate: number; n_starts: number }>;
  example_paths: Record<string, number[][]>;
}
interface E05Payload {
  beta_grid: number[];
  gamma_grid: number[];
  theta_true: number[];
  surfaces: Record<string, { surface: number[][]; cond_jtj: number }>;
}

const OPTIMIZER_COLORS: Record<string, string> = {
  grid_refined_gauss_newton: "var(--solver-rk4)",
  gauss_newton_log: "var(--accent)",
  levenberg_marquardt: "var(--good)",
  nelder_mead: "var(--warn)",
  coordinate_descent_golden: "var(--c-E)",
};

function Heatmap({ betaGrid, gammaGrid, surface, width = 480, height = 400 }: { betaGrid: number[]; gammaGrid: number[]; surface: number[][]; width?: number; height?: number }) {
  const flat = surface.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const n2 = surface.length;
  const n1 = surface[0]?.length ?? 0;
  const cellW = width / n1;
  const cellH = height / n2;

  // A small viridis-like sequential stop list (perceptually ordered, not a
  // rainbow hue-sweep -- see the dataviz skill's palette guidance) used to
  // color the cost surface: dark desaturated blue (low cost, the basin we
  // want to draw the eye to) through teal/green to pale yellow (high cost).
  const STOPS: [number, number, number][] = [
    [22, 15, 60], // deep indigo
    [40, 60, 110],
    [30, 110, 130],
    [60, 160, 120],
    [160, 200, 100],
    [240, 230, 140], // pale yellow
  ];
  function lerp(a: number, b: number, t: number) {
    return a + (b - a) * t;
  }
  function color(v: number): string {
    const t = max > min ? (Math.log(v + 1) - Math.log(min + 1)) / (Math.log(max + 1) - Math.log(min + 1)) : 0;
    const scaled = Math.min(Math.max(t, 0), 1) * (STOPS.length - 1);
    const i0 = Math.floor(scaled);
    const i1 = Math.min(i0 + 1, STOPS.length - 1);
    const f = scaled - i0;
    const [r0, g0, b0] = STOPS[i0];
    const [r1, g1, b1] = STOPS[i1];
    const r = Math.round(lerp(r0, r1, f));
    const g = Math.round(lerp(g0, g1, f));
    const b = Math.round(lerp(b0, b1, f));
    return `rgb(${r}, ${g}, ${b})`;
  }

  return (
    <svg width={width} height={height}>
      {surface.map((row, i2) =>
        row.map((v, i1) => <rect key={`${i1}-${i2}`} x={i1 * cellW} y={height - (i2 + 1) * cellH} width={cellW + 0.5} height={cellH + 0.5} fill={color(v)} />)
      )}
    </svg>
  );
}

export default function FittingStudio() {
  const e06 = useExperiment<E06Payload>("E06");
  const e05 = useExperiment<E05Payload>("E05");
  const [sigmaKey, setSigmaKey] = useState("0.05");

  const surfaceEntry = e05.data?.surfaces[sigmaKey] ?? e05.data?.surfaces[Object.keys(e05.data?.surfaces ?? {})[0]];

  return (
    <div>
      <PageHeader eyebrow="04 · Precomputed (E05/E06)" title="Fitting Studio">
        Watch five optimizers descend the same J(β, γ) cost landscape from noisy synthetic data. Nelder-Mead's
        crawling simplex versus Gauss-Newton and Levenberg-Marquardt's few long jumps is a genuinely visible
        difference on the same contours.
      </PageHeader>

      <div className="grid-2">
        <Card>
          <SectionTitle sub={e05.data ? `κ(JᵀJ) ≈ ${surfaceEntry?.cond_jtj.toFixed(1)}` : undefined}>Cost landscape J(β, γ)</SectionTitle>
          {e05.loading && <p>Loading…</p>}
          {e05.data && surfaceEntry && (
            <>
              <div className="pill-row" style={{ marginBottom: 10 }}>
                {Object.keys(e05.data.surfaces).map((k) => (
                  <button key={k} className={"btn" + (k === sigmaKey ? " active" : "")} onClick={() => setSigmaKey(k)}>
                    σ = {(parseFloat(k) * 100).toFixed(0)}% of max(I)
                  </button>
                ))}
              </div>
              <Heatmap betaGrid={e05.data.beta_grid} gammaGrid={e05.data.gamma_grid} surface={surfaceEntry.surface} />
              <p style={{ fontSize: "var(--fs-xs)", color: "var(--text-tertiary)" }}>
                x: β ∈ [{e05.data.beta_grid[0].toFixed(2)}, {e05.data.beta_grid.at(-1)!.toFixed(2)}] · y: γ ∈ [
                {e05.data.gamma_grid[0].toFixed(2)}, {e05.data.gamma_grid.at(-1)!.toFixed(2)}] · darker = lower cost
              </p>
            </>
          )}
        </Card>

        <Card>
          <SectionTitle sub="each optimizer's path from the same noisy dataset & start point">Optimizer paths</SectionTitle>
          {e06.loading && <p>Loading…</p>}
          {e06.data && (
            <LineChart
              width={480}
              height={400}
              xLabel="β"
              yLabel="γ"
              series={Object.entries(e06.data.example_paths).map(([name, path]) => ({
                id: name,
                color: OPTIMIZER_COLORS[name] ?? "var(--accent)",
                points: path.map((p) => ({ x: p[0], y: p[1] })),
              }))}
              markers={[{ x: e06.data.theta_true[0], y: e06.data.theta_true[1], color: "var(--text-primary)", label: "truth" }]}
              formatX={(v) => v.toFixed(2)}
              formatY={(v) => v.toFixed(2)}
            />
          )}
        </Card>
      </div>

      <Card>
        <SectionTitle sub="mean over many random starts (E06)">Optimizer shoot-out</SectionTitle>
        {e06.data && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 16 }}>
            {Object.entries(e06.data.results).map(([name, r]) => (
              <div key={name}>
                <div style={{ fontSize: "var(--fs-xs)", color: OPTIMIZER_COLORS[name], fontWeight: 700, marginBottom: 6 }}>{name.replace(/_/g, " ")}</div>
                <Stat label="success rate" value={`${(r.success_rate * 100).toFixed(0)}%`} />
                <Stat label="mean iterations" value={r.mean_iterations?.toFixed(1) ?? "—"} />
                <Stat label="mean f-evals" value={r.mean_fevals?.toFixed(0) ?? "—"} />
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
