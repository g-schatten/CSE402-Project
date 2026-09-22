"""Content-hash-based manifest so reruns with an unchanged config are
skipped, and every result is traceable to (experiment id, config hash, git
SHA, library version, seed, wall time, host) -- PLAN.md section 4.7 / 9.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from sirlab import __version__
from sirlab.io.schema import ArtifactManifestEntry, dump_json


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class ManifestWriter:
    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.entries: dict[str, dict] = {}
        if self.manifest_path.exists():
            self.entries = json.loads(self.manifest_path.read_text())

    def should_skip(self, experiment_id: str, config: dict, *, force: bool = False) -> bool:
        if force:
            return False
        prior = self.entries.get(experiment_id)
        return prior is not None and prior.get("config_hash") == config_hash(config)

    def record(self, experiment_id: str, config: dict, *, seed: int, wall_time_s: float, output_files: list[str]) -> None:
        entry = ArtifactManifestEntry(
            experiment_id=experiment_id,
            config_hash=config_hash(config),
            git_sha=git_sha(),
            library_version=__version__,
            seed=seed,
            wall_time_s=wall_time_s,
            host=platform.node(),
            output_files=output_files,
        )
        self.entries[experiment_id] = entry.__dict__
        self._save()

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as fh:
            json.dump(self.entries, fh, indent=2)
