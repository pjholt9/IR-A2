from pathlib import Path

import trimesh
from spatialgeometry import Mesh

# Keep model paths relative to this module so it works from any launch directory.
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "part_models"

# The STLs are in millimetres and are much larger than the buckets, so shrink them as well
# Convert the source STL dimensions to metres before applying the display scale.
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


# Cache a centred STL copy so its mesh origin is convenient for grasp poses.
def _centred_stl(filename):
    """Write (once) a copy of the STL whose bounding-box centre is at the origin and return its path."""
    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / filename
    if not out.exists():
        mesh = trimesh.load(str(ROOT / filename))
                # Move the bounding-box centre to the origin so the object pose is intuitive.
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        mesh.export(str(out))
    return out


# Build the matching coloured mesh at the supplied world pose.
def make_part(colour, pose):
    """Create the sled body (brown) or barrel (black) mesh centred on pose."""
        # Apply unit conversion and the extra shrink factor together for these large models.
    s = MM_TO_M * PART_SCALE
    mesh = Mesh(
        filename=str(_centred_stl(PART_FILES[colour])),
        scale=[s, s, s],
        pose=pose,
        color=PART_COLOURS[colour],
    )
    mesh._collision = False
    return mesh
