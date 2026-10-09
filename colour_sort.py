import colorsys
import os
import random

import numpy as np

BROWN = "brown"
BLACK = "black"

# Reference colours (0-1 RGB) used when the HSV rules are inconclusive
_REFERENCE = {
    BLACK: np.array([0.05, 0.05, 0.05]),
    BROWN: np.array([0.40, 0.26, 0.13]),
}

BLACK_MAX_VALUE = 0.20
BROWN_HUE_RANGE_DEG = (10.0, 45.0)
BROWN_MIN_SATURATION = 0.30


def _to_unit_rgb(rgb):
    """Accept an RGB triple in 0-255 or 0-1 and return it as 0-1 floats."""
    rgb = np.asarray(rgb, dtype=float).reshape(-1)[:3]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def classify_colour(rgb):
    """Return "brown" or "black" for a single RGB colour."""
    r, g, b = _to_unit_rgb(rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    if v < BLACK_MAX_VALUE:
        return BLACK
    if BROWN_HUE_RANGE_DEG[0] <= h * 360.0 <= BROWN_HUE_RANGE_DEG[1] and s >= BROWN_MIN_SATURATION:
        return BROWN

    rgb_arr = np.array([r, g, b])
    return min(_REFERENCE, key=lambda name: np.linalg.norm(rgb_arr - _REFERENCE[name]))


def classify_image(image, mask=None):
    """Classify an object from an (H, W, 3) image using the median colour of its pixels."""
    pixels = np.asarray(image, dtype=float)
    pixels = pixels[mask] if mask is not None else pixels.reshape(-1, pixels.shape[-1])
    return classify_colour(np.median(pixels[:, :3], axis=0))


def classify_image_file(path):
    """Load an image file and classify its dominant colour."""
    from PIL import Image

    with Image.open(path) as img:
        return classify_image(np.asarray(img.convert("RGB")))


class SimulatedCamera:
    """Stand-in for a real camera: when the object flag is raised, 'captures' one of the image files at random."""

    def __init__(self, folder=None, filenames=("black.png", "brown.png")):
        folder = folder or os.path.dirname(os.path.abspath(__file__))
        self.image_paths = [os.path.join(folder, name) for name in filenames]
        self.object_present = False
        self.last_rgb = None
        self._fig = None

    def capture(self):
        """Return the path of a randomly selected image, as if photographing the object."""
        return random.choice(self.image_paths)

    def detect_colour(self):
        """If the object flag is set, capture an image, show it, and return its classified colour, else None."""
        if not self.object_present:
            return None
        from PIL import Image

        path = self.capture()
        with Image.open(path) as img:
            pixels = np.asarray(img.convert("RGB"))
        colour = classify_image(pixels)
        self.last_rgb = tuple(float(c) for c in np.median(pixels.reshape(-1, 3), axis=0) / 255.0)
        print(f"Camera image {os.path.basename(path)} -> {colour}")
        self.show_image(pixels, f"Camera: {os.path.basename(path)} -> {colour}")
        return colour

    def show_image(self, pixels, title):
        """Open a non-blocking window displaying the captured image."""
        import matplotlib.pyplot as plt

        self.close_image()
        self._fig = plt.figure("Camera")
        plt.imshow(pixels)
        plt.title(title)
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.05)

    def close_image(self):
        """Close the image window if it is open."""
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None

    def pump(self):
        """Keep the image window responsive while the simulation runs."""
        if self._fig is not None:
            self._fig.canvas.flush_events()


if __name__ == "__main__":
    samples = {
        "black": (10, 10, 12),
        "dark grey": (40, 40, 40),
        "brown": (101, 67, 33),
        "light brown": (150, 100, 60),
        "dark brown": (60, 35, 20),
    }
    for name, rgb in samples.items():
        print(f"{name:12s} {rgb} -> {classify_colour(rgb)}")
