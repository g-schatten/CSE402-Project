"""Dataclasses mirroring the JSON artifact contracts every experiment
writes, plus a to_json/from_json pair that handles numpy arrays. Kept
deliberately simple (dataclasses + asdict) rather than a heavier validation
framework, since the whole point is that every consumer (notebooks, the web
dashboard, the report figures) reads exactly this shape.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


@dataclass
class ArtifactManifestEntry:
    experiment_id: str
    config_hash: str
    git_sha: str
    library_version: str
    seed: int
    wall_time_s: float
    host: str
    output_files: list[str] = field(default_factory=list)


def to_json_dict(obj) -> dict:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def dump_json(obj, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(to_json_dict(obj), fh, cls=NumpyJSONEncoder, indent=2)


def dumps_json(obj) -> str:
    return json.dumps(to_json_dict(obj), cls=NumpyJSONEncoder, indent=2)
