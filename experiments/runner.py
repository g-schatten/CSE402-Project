"""CLI entry point: `python -m experiments.runner E07 [--profile quick|full]
[--force]` runs one experiment; `--all` runs the whole registry in id order.
Every run is content-hash cached via ManifestWriter unless --force is given.
"""
from __future__ import annotations

import argparse
import time

from experiments.common import load_config
from experiments.registry import REGISTRY, load_experiment_module
from sirlab.io.export import export_result
from sirlab.io.manifest import ManifestWriter

MANIFEST_PATH = "results/manifest.json"


def run_one(exp_id: str, *, profile: str = "quick", force: bool = False) -> None:
    spec = REGISTRY[exp_id]
    config = load_config(spec.config_file)
    config["_profile"] = profile
    manifest = ManifestWriter(MANIFEST_PATH)
    if manifest.should_skip(exp_id, config, force=force):
        print(f"[{exp_id}] up to date (config hash unchanged); skipping. Use --force to rerun.")
        return

    module = load_experiment_module(spec)
    print(f"[{exp_id}] {spec.name} -- running (profile={profile})...")
    t0 = time.time()
    payload = module.run(config)
    wall = time.time() - t0
    out_path = export_result(exp_id, payload)
    seed = config.get("seed", 0)
    manifest.record(exp_id, config, seed=seed, wall_time_s=wall, output_files=[str(out_path)])
    print(f"[{exp_id}] done in {wall:.1f}s -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run SIRLab experiments")
    parser.add_argument("experiment", nargs="?", help="Experiment id, e.g. E07")
    parser.add_argument("--all", action="store_true", help="Run every experiment in the registry")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.all:
        for exp_id in REGISTRY:
            run_one(exp_id, profile=args.profile, force=args.force)
    elif args.experiment:
        run_one(args.experiment.upper(), profile=args.profile, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
