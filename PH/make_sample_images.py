import os

import numpy as np
from PIL import Image

# Generate samples beside this script so the simulated camera can find them by default.
FOLDER = os.path.dirname(os.path.abspath(__file__))
# These RGB values give the colour sorter one known black and one known brown part image.
SAMPLES = {"black.png": (14, 14, 16), "brown.png": (101, 67, 33)}


# Add repeatable pixel noise so sample images look less like perfectly flat test swatches.
def make_image(rgb, size=(200, 200), noise=8, seed=0):
    """Create a noisy solid-colour image to stand in for a camera photo."""
    rng = np.random.default_rng(seed)
    base = np.ones((size[1], size[0], 3)) * np.array(rgb)
    img = np.clip(base + rng.normal(0, noise, base.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(img)


# Regenerate both camera test images when this helper is run directly.
if __name__ == "__main__":
    for name, rgb in SAMPLES.items():
        make_image(rgb).save(os.path.join(FOLDER, name))
        print(f"Wrote {name}")
