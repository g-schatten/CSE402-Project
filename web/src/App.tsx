import { NavLink, Route, Routes } from "react-router-dom";
import { useThemeStore } from "./state/theme";

import Overview from "./pages/Overview";
import ModelLab from "./pages/ModelLab";
import SolverArena from "./pages/SolverArena";
import Stability from "./pages/Stability";
import FittingStudio from "./pages/FittingStudio";
import Uncertainty from "./pages/Uncertainty";
import ExperimentExplorer from "./pages/ExperimentExplorer";
import Crossover from "./pages/Crossover";
import RealData from "./pages/RealData";
import Methods from "./pages/Methods";

const NAV = [
  { to: "/", label: "Overview", num: "00" },
  { to: "/model-lab", label: "Model Lab", num: "01" },
  { to: "/solver-arena", label: "Solver Arena", num: "02" },
  { to: "/stability", label: "Stability", num: "03" },
  { to: "/fitting-studio", label: "Fitting Studio", num: "04" },
  { to: "/uncertainty", label: "Uncertainty", num: "05" },
  { to: "/explorer", label: "Experiment Explorer", num: "06" },
  { to: "/crossover", label: "Crossover σ*", num: "07" },
  { to: "/real-data", label: "Real Data", num: "08" },
  { to: "/methods", label: "Methods & Team", num: "09" },
];

export default function App() {
  const { theme, toggle } = useThemeStore();

  return (
    <div className="app-shell">
      <nav className="rail">
        <div className="rail__brand">
          SIRLab <small>CSE 402 · Group 2</small>
        </div>
        <div className="rail__nav">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => "rail__link" + (isActive ? " active" : "")}
            >
              <span className="rail__num">{item.num}</span>
              {item.label}
            </NavLink>
          ))}
        </div>
        <button className="theme-toggle" onClick={toggle}>
          {theme === "dark" ? "☾ Dark" : "☀ Light"}
        </button>
      </nav>
      <main className="content">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/model-lab" element={<ModelLab />} />
          <Route path="/solver-arena" element={<SolverArena />} />
          <Route path="/stability" element={<Stability />} />
          <Route path="/fitting-studio" element={<FittingStudio />} />
          <Route path="/uncertainty" element={<Uncertainty />} />
          <Route path="/explorer" element={<ExperimentExplorer />} />
          <Route path="/crossover" element={<Crossover />} />
          <Route path="/real-data" element={<RealData />} />
          <Route path="/methods" element={<Methods />} />
        </Routes>
      </main>
    </div>
  );
}
