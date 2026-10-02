import { PageHeader, Card, SectionTitle, Stat, Legend } from "../components/ui";
import { useExperiment } from "../state/useExperiment";
import { LineChart } from "../charts/LineChart";

interface E14Payload {
  n_total: number;
  i0: number;
  days: number[];
  s_observed: number[];
  results: Record<string, { theta_hat: number[]; r0: number; converged: boolean; predicted_s: number[]; residuals: number[] }>;
}

const SOLVER_COLOR: Record<string, string> = { rk4: "var(--solver-rk4)", euler: "var(--solver-euler)" };

export default function RealData() {
  const { data, loading, error } = useExperiment<E14Payload>("E14");

  return (
    <div>
      <PageHeader eyebrow="08 · Precomputed (E14) · Optional extension" title="Real Data: Eyam, 1666">
        The same estimation pipeline applied to the historical Eyam plague outbreak (Raggett, 1982; source details in
        <code> data/raw/provenance.md</code>). This is a demonstration, not a proof.
      </PageHeader>

      {loading && <Card>Loading…</Card>}
      {error && <Card>Could not load E14 ({error}). This experiment is optional and may not have been run.</Card>}

      {data && (
        <>
          <Card>
            <SectionTitle sub={`N = ${data.n_total}, I₀ = ${data.i0}`}>Susceptible count over time: fit vs observed</SectionTitle>
            <LineChart
              width={1080}
              height={340}
              xLabel="days after 18 June 1666"
              yLabel="S(t)"
              series={[
                { id: "observed", color: "var(--text-primary)", points: data.days.map((d, i) => ({ x: d, y: data.s_observed[i] })) },
                ...Object.entries(data.results).map(([solver, r]) => ({
                  id: `fit (${solver})`,
                  color: SOLVER_COLOR[solver] ?? "var(--accent)",
                  points: data.days.map((d, i) => ({ x: d, y: r.predicted_s[i] })),
                })),
              ]}
              formatX={(v) => v.toFixed(0)}
              formatY={(v) => v.toFixed(0)}
            />
            <Legend items={[{ label: "observed", color: "var(--text-primary)" }, ...Object.keys(data.results).map((s) => ({ label: `fit (${s})`, color: SOLVER_COLOR[s] ?? "var(--accent)" }))]} />
          </Card>

          <div className="grid-2">
            {Object.entries(data.results).map(([solver, r]) => (
              <Card key={solver}>
                <SectionTitle>Solver: {solver}</SectionTitle>
                <div className="grid-3">
                  <Stat label="β̂" value={r.theta_hat[0].toFixed(4)} />
                  <Stat label="γ̂" value={r.theta_hat[1].toFixed(4)} />
                  <Stat label="R̂₀ = β̂/γ̂" value={r.r0.toFixed(2)} />
                </div>
              </Card>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
