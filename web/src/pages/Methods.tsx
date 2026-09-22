import { PageHeader, Card, SectionTitle } from "../components/ui";
import { Katex } from "../components/Katex";

const TEAM = [
  { name: "Saif Uz Zaman", role: "Numerics core lead" },
  { name: "Sayaad Muzahid Masfi", role: "Report, deck & presenter" },
  { name: "Ibtida bin Ahmed", role: "Estimation & UQ lead" },
  { name: "Sakif Naieb Raiyan", role: "Web engineering lead" },
  { name: "Aurchi Chowdhury", role: "Experiments & data lead" },
];

export default function Methods() {
  return (
    <div>
      <PageHeader eyebrow="09 · Reference" title="Methods & Team">
        The full equation set behind every page in this dashboard, and the people who built it.
      </PageHeader>

      <Card>
        <SectionTitle>The SIR model</SectionTitle>
        <Katex block tex="\dfrac{dS}{dt} = -\dfrac{\beta S I}{N},\quad \dfrac{dI}{dt} = \dfrac{\beta S I}{N} - \gamma I,\quad \dfrac{dR}{dt} = \gamma I" />
      </Card>

      <Card>
        <SectionTitle sub="reference.py">The semi-analytic gold standard</SectionTitle>
        <Katex block tex="\frac{dS}{dR} = -\frac{\beta}{\gamma N} S \;\;\Rightarrow\;\; S(t) = S_0 \exp\!\Big(-\frac{\beta}{\gamma N}\big(R(t)-R_0\big)\Big)" />
        <Katex block tex="t(R) = \int_0^{R} \frac{dr}{\gamma\big(N - r - S_0 e^{-\frac{\beta}{\gamma N} r}\big)}" />
        <p style={{ color: "var(--text-secondary)", fontSize: "var(--fs-sm)" }}>
          Evaluated by adaptive quadrature to machine precision, then inverted by a safeguarded Newton method
          (rtsafe) to obtain R(t), giving a reference trajectory with no ODE solver in the loop at all.
        </p>
      </Card>

      <Card>
        <SectionTitle>Solvers</SectionTitle>
        <p><strong>Forward Euler</strong> (order 1): <Katex tex="y_{n+1} = y_n + h f(t_n, y_n)" /></p>
        <p><strong>Heun</strong> (order 2): <Katex tex="y^* = y_n + h f(t_n,y_n),\quad y_{n+1} = y_n + \tfrac{h}{2}\big[f(t_n,y_n) + f(t_{n+1}, y^*)\big]" /></p>
        <p><strong>Classical RK4</strong> (order 4): <Katex tex="y_{n+1} = y_n + \tfrac{h}{6}(k_1+2k_2+2k_3+k_4)" /></p>
      </Card>

      <Card>
        <SectionTitle>Least-squares estimation</SectionTitle>
        <Katex block tex="J(\theta) = \sum_k w_k \big(\text{obs}_k - h(y(t_k;\theta))\big)^2" />
        <p style={{ color: "var(--text-secondary)", fontSize: "var(--fs-sm)" }}>
          Minimized by grid search, golden-section coordinate descent, Nelder-Mead, Gauss-Newton, and
          Levenberg-Marquardt — all hand-written, using the forward sensitivity equations (not finite differences)
          for every gradient-based method's Jacobian.
        </p>
      </Card>

      <Card>
        <SectionTitle>Base paper</SectionTitle>
        <p style={{ color: "var(--text-secondary)" }}>
          Capaldi, A., Behrend, S., Berman, B., Smith, J., Wright, J., &amp; Lloyd, A. L. (2012). Parameter
          estimation and uncertainty quantification for an epidemic model.{" "}
          <em>Mathematical Biosciences and Engineering</em>, 9(3), 553–576.{" "}
          <a href="https://doi.org/10.3934/mbe.2012.9.553" style={{ color: "var(--accent-strong)" }}>
            doi.org/10.3934/mbe.2012.9.553
          </a>
        </p>
        <p style={{ color: "var(--text-secondary)" }}>
          Supporting reference: Prodanov, D. (2021). Analytical parameterization of the SIR model...{" "}
          <em>Entropy</em>, 23(1), 59.
        </p>
      </Card>

      <Card>
        <SectionTitle>Team — CSE 402, Section B, Group 2</SectionTitle>
        <div className="grid-3">
          {TEAM.map((t) => (
            <div key={t.name}>
              <div style={{ fontWeight: 600 }}>{t.name}</div>
              <div style={{ color: "var(--text-secondary)", fontSize: "var(--fs-sm)" }}>{t.role}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
