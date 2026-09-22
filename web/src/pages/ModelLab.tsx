import { useMemo, useState } from "react";
import { PageHeader, Card, SectionTitle, Slider, Legend, Stat } from "../components/ui";
import { MODEL_REGISTRY, type Model } from "../numerics/models";
import { integrateRK4 } from "../numerics/solvers";
import { SIRReference } from "../numerics/reference";
import { LineChart } from "../charts/LineChart";

const N = 1000;
const COMPARTMENT_COLOR: Record<string, string> = { S: "var(--c-S)", E: "var(--c-E)", I: "var(--c-I)", R: "var(--c-R)", V: "var(--c-V)" };

export default function ModelLab() {
  const [modelName, setModelName] = useState<keyof typeof MODEL_REGISTRY>("SIR");
  const [beta, setBeta] = useState(0.3);
  const [gamma, setGamma] = useState(0.1);
  const [sigmaInc, setSigmaInc] = useState(0.2);
  const [xi, setXi] = useState(0.02);
  const [nu, setNu] = useState(0.01);
  const [i0, setI0] = useState(1);
  const [tFinal, setTFinal] = useState(120);

  const model: Model = useMemo(() => new (MODEL_REGISTRY[modelName] as any)(N), [modelName]);
  const theta = useMemo(() => {
    if (modelName === "SIR") return [beta, gamma];
    if (modelName === "SEIR") return [beta, sigmaInc, gamma];
    if (modelName === "SIRS") return [beta, gamma, xi];
    return [beta, gamma, nu];
  }, [modelName, beta, gamma, sigmaInc, xi, nu]);

  const y0 = useMemo(() => model.initialState({ i0, e0: 2, v0: 0 }), [model, i0]);
  const traj = useMemo(() => integrateRK4(model, y0, theta, 0, tFinal, Math.max(tFinal / 800, 0.02)), [model, y0, theta, tFinal]);

  const r0 = model.r0(theta);
  const sIdx = model.stateNames.indexOf("S");
  const iIdx = model.stateNames.indexOf("I");

  const ref = useMemo(() => (modelName === "SIR" ? new SIRReference({ beta, gamma, nTotal: N, s0: y0[sIdx], i0: y0[iIdx] }) : null), [
    modelName,
    beta,
    gamma,
    y0,
    sIdx,
    iIdx,
  ]);

  const seriesByCompartment = model.stateNames.map((name, idx) => ({
    id: name,
    color: COMPARTMENT_COLOR[name] ?? "var(--accent)",
    points: traj.t.map((t, k) => ({ x: t, y: traj.y[k][idx] })),
  }));

  const phasePoints = traj.t.map((_, k) => ({ x: traj.y[k][sIdx], y: traj.y[k][iIdx] }));
  const phaseInvariantPoints = useMemo(() => {
    if (!ref) return [];
    const rGrid: { x: number; y: number }[] = [];
    const rMax = ref.rInfinity * 0.999;
    for (let k = 0; k <= 200; k++) {
      const r = (k / 200) * rMax;
      rGrid.push({ x: ref.sOfR(r), y: N - r - ref.sOfR(r) });
    }
    return rGrid;
  }, [ref]);

  const rEffPoints = traj.t.map((t, k) => ({ x: t, y: r0 * (traj.y[k][sIdx] / N) }));

  return (
    <div>
      <PageHeader eyebrow="01 · Interactive" title="Model Lab">
        Choose a model variant and drag its parameters. Three linked views: the time series, the (S, I) phase
        portrait against the exact SIR invariant curve, and the effective reproduction number R_eff(t) with the
        epidemic threshold at 1.
      </PageHeader>

      <Card>
        <div className="control-row">
          <div className="control">
            <label>
              <span>Model</span>
            </label>
            <select value={modelName} onChange={(e) => setModelName(e.target.value as keyof typeof MODEL_REGISTRY)}>
              {Object.keys(MODEL_REGISTRY).map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </div>
          <Slider label="β" value={beta} min={0.05} max={1.2} step={0.01} onChange={setBeta} />
          <Slider label="γ" value={gamma} min={0.02} max={0.6} step={0.01} onChange={setGamma} />
          {modelName === "SEIR" && <Slider label="σ (incubation)" value={sigmaInc} min={0.05} max={1.0} step={0.01} onChange={setSigmaInc} />}
          {modelName === "SIRS" && <Slider label="ξ (waning)" value={xi} min={0} max={0.1} step={0.002} onChange={setXi} />}
          {modelName === "SIR-V" && <Slider label="ν (vaccination)" value={nu} min={0} max={0.05} step={0.001} onChange={setNu} />}
          <Slider label="I₀" value={i0} min={1} max={50} step={1} onChange={setI0} />
          <Slider label="T (days)" value={tFinal} min={30} max={300} step={5} onChange={setTFinal} />
        </div>
      </Card>

      <div className="grid-2">
        <Card>
          <SectionTitle sub="RK4 solve, decimated for display">Time series</SectionTitle>
          <LineChart width={520} height={300} xLabel="t (days)" yLabel="people" series={seriesByCompartment} formatX={(v) => v.toFixed(0)} formatY={(v) => v.toFixed(0)} />
          <Legend items={model.stateNames.map((n) => ({ label: n, color: COMPARTMENT_COLOR[n] ?? "var(--accent)" }))} />
        </Card>
        <Card>
          <SectionTitle sub={modelName === "SIR" ? "dashed = exact phase invariant (eq. 2.1)" : "phase invariant is SIR-only"}>
            Phase portrait (S, I)
          </SectionTitle>
          <LineChart
            width={520}
            height={300}
            xLabel="S"
            yLabel="I"
            series={[
              { id: "trajectory", color: "var(--c-I)", points: phasePoints },
              ...(phaseInvariantPoints.length ? [{ id: "exact invariant", color: "var(--text-tertiary)", points: phaseInvariantPoints, dashed: true }] : []),
            ]}
            formatX={(v) => v.toFixed(0)}
            formatY={(v) => v.toFixed(0)}
          />
        </Card>
      </div>

      <Card>
        <SectionTitle sub="R_eff(t) = R0 · S(t)/N — epidemic grows while this is above 1">Effective reproduction number</SectionTitle>
        <LineChart
          width={1080}
          height={220}
          xLabel="t (days)"
          yLabel="R_eff"
          series={[{ id: "R_eff", color: "var(--accent)", points: rEffPoints }]}
          markers={[]}
          formatX={(v) => v.toFixed(0)}
          formatY={(v) => v.toFixed(2)}
        />
        <div className="grid-3" style={{ marginTop: 12 }}>
          <Stat label="R₀ = β/γ" value={r0.toFixed(2)} color={r0 > 1 ? "var(--bad)" : "var(--good)"} />
          {ref && <Stat label="Analytic peak I" value={ref.peakI.toFixed(1)} />}
          {ref && <Stat label="Analytic peak time" value={`${ref.peakTime().toFixed(1)} d`} />}
        </div>
      </Card>
    </div>
  );
}
