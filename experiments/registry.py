"""Experiment registry (PLAN.md section 5): maps experiment ids to their
runner functions and default config paths. `python -m experiments.runner E07`
looks an experiment up here.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass
class ExperimentSpec:
    id: str
    name: str
    module: str  # experiments.<module>, must expose run(config: dict) -> dict
    config_file: str


REGISTRY: dict[str, ExperimentSpec] = {
    "E01": ExperimentSpec("E01", "Convergence & order", "E01_convergence", "e01.yaml"),
    "E02": ExperimentSpec("E02", "Invariant drift", "E02_invariants", "e02.yaml"),
    "E03": ExperimentSpec("E03", "Stability frontier", "E03_stability", "e03.yaml"),
    "E04": ExperimentSpec("E04", "Structural accuracy", "E04_structural", "e04.yaml"),
    "E05": ExperimentSpec("E05", "Cost landscape", "E05_landscape", "e05.yaml"),
    "E06": ExperimentSpec("E06", "Optimizer shoot-out", "E06_optimizers", "e06.yaml"),
    "E07": ExperimentSpec("E07", "Solver-in-the-loop recovery", "E07_solver_recovery", "e07.yaml"),
    "E08": ExperimentSpec("E08", "Noise degradation", "E08_noise", "e08.yaml"),
    "E09": ExperimentSpec("E09", "Sampling & window", "E09_sampling", "e09.yaml"),
    "E10": ExperimentSpec("E10", "Initial conditions", "E10_initial_conditions", "e10.yaml"),
    "E11": ExperimentSpec("E11", "Uncertainty quantification", "E11_uq", "e11.yaml"),
    "E12": ExperimentSpec("E12", "Model variants", "E12_variants", "e12.yaml"),
    "E13": ExperimentSpec("E13", "Crossover sigma*", "E13_crossover", "e13.yaml"),
    "E14": ExperimentSpec("E14", "Real data", "E14_realdata", "e14.yaml"),
}


def load_experiment_module(spec: ExperimentSpec):
    return importlib.import_module(f"experiments.{spec.module}")
