import { useState } from "react";
import { PageHeader, Card, SectionTitle, Stat, Legend } from "../components/ui";
import { useExperiment } from "../state/useExperiment";
import { LineChart } from "../charts/LineChart";

interface E11Payload {
  theta_true: number[];
  point_estimate: number[];
  asymptotic: { ellipse: number[][]; chi2: number; rho: number; cov: number[][] };
  bootstrap: { thetas: number[][]; mean: number[]; std: number[] };
  montecarlo: { bias: number[]; variance: number[]; rmse: number[]; n_converged: number };
  mcmc: { pooled_samples: number[][]; rhat: number[]; ess_beta: number; ess_gamma: number; acceptance_rate: number[]; posterior_mean: number[] };
  profile_likelihood: { beta_grid: number[]; cost_full_window: number[]; cost_truncated_window: number[]; t_peak: number; t_truncated: number };
}

const LAYERS = [
  { id: "asymptotic", label: "Asymptotic ellipse", color: "var(--accent)" },
  { id: "bootstrap", label: "Bootstrap cloud", color: "var(--warn)" },
  { id: "mcmc", label: "MCMC posterior", color: "var(--good)" },
];

export default function Uncertainty() {
  const { data, loading, error } = useExperiment<E11Payload>("E11");
  const [visible, setVisible] = useState<Record<string, boolean>>({ asymptotic: true, bootstrap: true, mcmc: true });

  return (
    <div>
      <PageHeader eyebrow="05 · Precomputed (E11)" title="Uncertainty Quantification">
        Four independent routes to a confidence region for (β, γ), overlaid on one figure: asymptotic linearization,
        residual bootstrap, Monte Carlo over fresh noise realizations (the only one that knows the ground truth), and
        adaptive Metropolis MCMC with Gelman-Rubin R̂ across four chains.
      </PageHeader>

      {loading && <Card>Loading…</Card>}
      {error && <Card>Could not load E11 results ({error}). Run the experiment first.</Card>}

      {data && (
        <>
          <Card>
            <SectionTitle sub="toggle layers">Overlay</SectionTitle>
            <div className="pill-row" style={{ marginBottom: 12 }}>
              {LAYERS.map((l) => (
                <button key={l.id} className={"btn" + (visible[l.id] ? " active" : "")} onClick={() => setVisible((v) => ({ ...v, [l.id]: !v[l.id] }))}>
                  {l.label}
                </button>
              ))}
            </div>
            <LineChart
              width={640}
              height={420}
              xLabel="β"
              yLabel="γ"
              series={[
                ...(visible.asymptotic ? [{ id: "asymptotic 95%", color: "var(--accent)", points: data.asymptotic.ellipse.map((p) => ({ x: p[0], y: p[1] })) }] : []),
                ...(visible.bootstrap
                  ? [
                      {
                        id: "bootstrap",
                        color: "var(--warn)",
                        points: data.bootstrap.thetas.map((p) => ({ x: p[0], y: p[1] })),
                        scatter: true,
                      },
                    ]
                  : []),
                ...(visible.mcmc
                  ? [
                      {
                        id: "mcmc posterior",
                        color: "var(--good)",
                        points: data.mcmc.pooled_samples.map((p) => ({ x: p[0], y: p[1] })),
                        scatter: true,
                      },
                    ]
                  : []),
              ]}
              markers={[{ x: data.theta_true[0], y: data.theta_true[1], color: "var(--text-primary)", label: "truth" }]}
              formatX={(v) => v.toFixed(3)}
              formatY={(v) => v.toFixed(3)}
            />
            <Legend items={[...LAYERS, { label: "truth", color: "var(--text-primary)" }]} />
          </Card>

          <div className="grid-3">
            <Card>
              <SectionTitle>Asymptotic</SectionTitle>
              <Stat label="ρ(β̂, γ̂)" value={data.asymptotic.rho.toFixed(3)} />
              <Stat label="χ² (95%, 2 df)" value={data.asymptotic.chi2.toFixed(3)} />
            </Card>
            <Card>
              <SectionTitle>Monte Carlo (knows truth)</SectionTitle>
              <Stat label="bias(β)" value={data.montecarlo.bias[0].toExponential(2)} />
              <Stat label="bias(γ)" value={data.montecarlo.bias[1].toExponential(2)} />
              <Stat label="RMSE" value={`${data.montecarlo.rmse[0].toFixed(4)}, ${data.montecarlo.rmse[1].toFixed(4)}`} />
            </Card>
            <Card>
              <SectionTitle>MCMC diagnostics</SectionTitle>
              <Stat label="R̂ (β, γ)" value={`${data.mcmc.rhat[0].toFixed(3)}, ${data.mcmc.rhat[1].toFixed(3)}`} color={Math.max(...data.mcmc.rhat) < 1.05 ? "var(--good)" : "var(--warn)"} />
              <Stat label="ESS (β, γ)" value={`${data.mcmc.ess_beta.toFixed(0)}, ${data.mcmc.ess_gamma.toFixed(0)}`} />
              <Stat label="mean acceptance rate" value={`${(100 * (data.mcmc.acceptance_rate.reduce((a, b) => a + b, 0) / data.mcmc.acceptance_rate.length)).toFixed(0)}%`} />
            </Card>
          </div>

          <Card>
            <SectionTitle sub="a flat profile is the visual signature of non-identifiability">Profile likelihood: full window vs pre-peak truncated window</SectionTitle>
            <LineChart
              width={1080}
              height={280}
              xLabel="β (γ profiled out)"
              yLabel="min cost"
              yLog
              series={[
                { id: `full window (T > peak @ ${data.profile_likelihood.t_peak.toFixed(0)}d)`, color: "var(--good)", points: data.profile_likelihood.beta_grid.map((b, i) => ({ x: b, y: data.profile_likelihood.cost_full_window[i] })) },
                { id: `truncated (T = ${data.profile_likelihood.t_truncated.toFixed(0)}d, pre-peak)`, color: "var(--bad)", points: data.profile_likelihood.beta_grid.map((b, i) => ({ x: b, y: data.profile_likelihood.cost_truncated_window[i] })) },
              ]}
              markers={[{ x: data.theta_true[0], y: 0, color: "var(--text-primary)" }]}
              formatX={(v) => v.toFixed(2)}
              formatY={(v) => v.toExponential(1)}
            />
          </Card>
        </>
      )}
    </div>
  );
}
