import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { PageHeader, Card, Stat, Legend } from "../components/ui";
import { SIR } from "../numerics/models";
import { integrateRK4 } from "../numerics/solvers";
import { LineChart } from "../charts/LineChart";

const N = 1000;
const BETA = 0.3;
const GAMMA = 0.1;

const RQ_CARDS = [
  {
    id: "RQ1",
    title: "Solver comparison",
    body: "How do Forward Euler, Heun, and RK4 differ in accuracy, convergence order, and numerical stability on the SIR system?",
    link: "/solver-arena",
  },
  {
    id: "RQ2",
    title: "Parameter recovery",
    body: "How does solver choice propagate into the accuracy of (β, γ) recovered via least-squares estimation on synthetic data?",
    link: "/fitting-studio",
  },
  {
    id: "RQ3",
    title: "Noise & sampling robustness",
    body: "How robust is parameter recovery to observation noise, sparse sampling, and initial conditions -- and does that hold across (β, γ) space?",
    link: "/crossover",
  },
];

export default function Overview() {
  const model = useMemo(() => new SIR(N), []);
  const [beta, setBeta] = useState(BETA);
  const traj = useMemo(() => integrateRK4(model, [N - 1, 1, 0], [beta, GAMMA], 0, 120, 0.1), [model, beta]);
  const r0 = beta / GAMMA;

  const seriesS = traj.t.map((t, i) => ({ x: t, y: traj.y[i][0] }));
  const seriesI = traj.t.map((t, i) => ({ x: t, y: traj.y[i][1] }));
  const seriesR = traj.t.map((t, i) => ({ x: t, y: traj.y[i][2] }));

  return (
    <div>
      <PageHeader eyebrow="CSE 402 · Numerical Analysis, Simulation & Modeling" title="Numerical Analysis of the SIR Epidemic Model">
        Solver accuracy, parameter estimation, and robustness to noisy observations — a controlled numerical study
        using the classical epidemic model as a testbed. Grounded in Capaldi et al. (2012); we make the ODE solver
        itself the central experimental variable.
      </PageHeader>

      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <Card>
          <div className="grid-2" style={{ alignItems: "center" }}>
            <div>
              <LineChart
                width={520}
                height={280}
                xLabel="t (days)"
                yLabel="people"
                series={[
                  { id: "S", color: "var(--c-S)", points: seriesS },
                  { id: "I", color: "var(--c-I)", points: seriesI },
                  { id: "R", color: "var(--c-R)", points: seriesR },
                ]}
                formatX={(v) => v.toFixed(0)}
                formatY={(v) => v.toFixed(0)}
              />
              <Legend
                items={[
                  { label: "Susceptible", color: "var(--c-S)" },
                  { label: "Infectious", color: "var(--c-I)" },
                  { label: "Recovered", color: "var(--c-R)" },
                ]}
              />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <div className="control">
                <label>
                  <span>β (transmission rate)</span>
                  <span className="value">{beta.toFixed(2)}</span>
                </label>
                <input type="range" min={0.05} max={0.9} step={0.01} value={beta} onChange={(e) => setBeta(parseFloat(e.target.value))} />
              </div>
              <div className="grid-2">
                <Stat label="R₀ = β/γ" value={r0.toFixed(2)} color={r0 > 1 ? "var(--bad)" : "var(--good)"} />
                <Stat label="γ (fixed)" value={GAMMA.toFixed(2)} />
              </div>
              <p style={{ color: "var(--text-secondary)", fontSize: "var(--fs-sm)", margin: 0 }}>
                dS/dt = −βSI/N &nbsp;·&nbsp; dI/dt = βSI/N − γI &nbsp;·&nbsp; dR/dt = γI
              </p>
            </div>
          </div>
        </Card>
      </motion.div>

      <div className="grid-3" style={{ marginTop: 20 }}>
        {RQ_CARDS.map((rq, i) => (
          <motion.a
            key={rq.id}
            href={`#${rq.link}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: 0.1 + i * 0.08 }}
            style={{ textDecoration: "none", color: "inherit" }}
          >
            <Card>
              <div style={{ color: "var(--accent-strong)", fontWeight: 700, fontSize: "var(--fs-xs)", letterSpacing: "0.06em" }}>
                {rq.id}
              </div>
              <div className="section-title" style={{ fontSize: "var(--fs-base)", marginTop: 4 }}>
                {rq.title}
              </div>
              <p style={{ color: "var(--text-secondary)", fontSize: "var(--fs-sm)" }}>{rq.body}</p>
            </Card>
          </motion.a>
        ))}
      </div>

      <Card>
        <div className="section-title">The headline finding</div>
        <p style={{ color: "var(--text-secondary)", maxWidth: "72ch" }}>
          There exists a <strong style={{ color: "var(--text-primary)" }}>crossover noise level σ*</strong> at which
          solver-induced bias in the recovered (β, γ) is exactly matched by noise-induced variance. Below σ*, the
          choice of ODE solver is the dominant error source and Euler at a coarse step is scientifically
          indefensible. Above σ*, Euler becomes statistically indistinguishable from RK4 — paying for a higher-order
          solver buys nothing. See the{" "}
          <a href="#/crossover" style={{ color: "var(--accent-strong)" }}>
            Crossover σ*
          </a>{" "}
          page for the measured curve.
        </p>
      </Card>
    </div>
  );
}
