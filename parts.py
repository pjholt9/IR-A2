from pathlib import Path

import trimesh
from spatialgeometry import Mesh

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "part_models"

# The STLs are in millimetres and are much larger than the buckets, so shrink them as well
MM_TO_M = 0.001
PART_SCALE = 0.3

PART_FILES = {
    "brown": "Sled Body Front.STL",
    "black": "Barrel.STL",
}
PART_COLOURS = {
    "brown": "saddlebrown",
    # Pure black renders pink in Swift, so use a near-black
    "black": "#0a0a0a",
}


def _centred_stl(filename):
    """Write (once) a copy of the STL whose bounding-box centre is at the origin and return its path."""
    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / filename
    if not out.exists():
        mesh = trimesh.load(str(ROOT / filename))
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        mesh.export(str(out))
    return out


def make_part(colour, pose):
    """Create the sled body (brown) or barrel (black) mesh centred on pose."""
    s = MM_TO_M * PART_SCALE
    return Mesh(
        filename=str(_centred_stl(PART_FILES[colour])),
        scale=[s, s, s],
        pose=pose,
        color=PART_COLOURS[colour],
    )
