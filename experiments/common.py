"""Shared helpers used across experiment scripts: config loading, model
construction from a config dict, and the reference-trajectory / forward-model
builders every experiment needs. Keeping this here avoids repeating the same
boilerplate in all fourteen experiment scripts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from sirlab.estimate.base import ForwardModel
from sirlab.models import REGISTRY as MODEL_REGISTRY
from sirlab.observe import prevalence
from sirlab.reference import SIRReference

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_config(filename: str) -> dict:
    with open(CONFIG_DIR / filename) as fh:
        return yaml.safe_load(fh)


def build_model(config: dict):
    model_name = config.get("model", "SIR")
    n_total = float(config["n_total"])
    return MODEL_REGISTRY[model_name](n_total), model_name


def build_y0(model, config: dict) -> np.ndarray:
    return model.initial_state(
        n=float(config["n_total"]),
        i0=float(config.get("i0", 1.0)),
        e0=float(config.get("e0", 0.0)),
        r0=float(config.get("r0", 0.0)),
        v0=float(config.get("v0", 0.0)),
    )


def build_theta(model, config: dict) -> np.ndarray:
    return np.array([float(config["theta"][name]) for name in model.param_names])


def sir_reference_from_config(config: dict) -> SIRReference:
    y0 = np.array([float(config["n_total"]) - float(config.get("i0", 1.0)), float(config.get("i0", 1.0)), 0.0])
    return SIRReference(
        beta=float(config["theta"]["beta"]),
        gamma=float(config["theta"]["gamma"]),
        n_total=float(config["n_total"]),
        s0=y0[0],
        i0=y0[1],
    )


def build_prevalence_forward_model(model, y0, t_obs, *, solver="rk4", h=None, rtol=1e-9, atol=1e-11, i_index=1):
    def obs_fn(result, t_obs):
        return prevalence(result, i_index=i_index, t_obs=t_obs)

    fwd = ForwardModel(model, y0, t_obs, obs_fn, solver=solver, h=h, rtol=rtol, atol=atol)
    fwd.obs_state_index = i_index
    return fwd


def quick_or_full(config: dict, key: str, *, profile: str):
    """Read config[key][profile] if present (config splits quick/full
    budgets), else config[key] directly."""
    val = config.get(key)
    if isinstance(val, dict) and profile in val:
        return val[profile]
    return val
