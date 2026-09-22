from sirlab.models.base import Model, ModelSpec
from sirlab.models.sir import SIR
from sirlab.models.seir import SEIR
from sirlab.models.sirs import SIRS
from sirlab.models.vaccination import SIRVaccination

REGISTRY = {
    "SIR": SIR,
    "SEIR": SEIR,
    "SIRS": SIRS,
    "SIR-V": SIRVaccination,
}

__all__ = ["Model", "ModelSpec", "SIR", "SEIR", "SIRS", "SIRVaccination", "REGISTRY"]
