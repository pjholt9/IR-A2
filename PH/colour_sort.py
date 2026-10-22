import colorsys
import os
import random

import numpy as np
from scipy import ndimage

BROWN = "brown"
BLACK = "black"

# Reference colours (0-1 RGB) used when the HSV rules are inconclusive
_REFERENCE = {
    BLACK: np.array([0.05, 0.05, 0.05]),
    BROWN: np.array([0.40, 0.26, 0.13]),
}

# These thresholds handle the common colours first; reference RGB values cover anything in between.
BLACK_MAX_VALUE = 0.20
BROWN_HUE_RANGE_DEG = (10.0, 45.0)
BROWN_MIN_SATURATION = 0.30
WHITE_BACKGROUND_MAX_SATURATION = 0.12
DARK_FOREGROUND_MAX_VALUE = 60
DISPLAY_PADDING_PX = 20


# Normalize camera values so the colour rules work with either 8-bit or unit RGB.
def _to_unit_rgb(rgb):
    """Accept an RGB triple in 0-255 or 0-1 and return it as 0-1 floats."""
    rgb = np.asarray(rgb, dtype=float).reshape(-1)[:3]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


# Use simple HSV rules for obvious dark/brown parts, then pick the closest reference colour.
def classify_colour(rgb):
    """Return "brown" or "black" for a single RGB colour."""
    r, g, b = _to_unit_rgb(rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    if v < BLACK_MAX_VALUE:
        return BLACK
    # Hue is returned as a fraction of a turn, so convert it to degrees before comparing.
    if BROWN_HUE_RANGE_DEG[0] <= h * 360.0 <= BROWN_HUE_RANGE_DEG[1] and s >= BROWN_MIN_SATURATION:
        return BROWN

    rgb_arr = np.array([r, g, b])
    return min(_REFERENCE, key=lambda name: np.linalg.norm(rgb_arr - _REFERENCE[name]))


# One median RGB sample keeps a few noisy pixels from changing the image's label.
def classify_image(image, mask=None):
    """Classify an object from an (H, W, 3) image using the median colour of its pixels."""
    pixels = np.asarray(image, dtype=float)
    pixels = pixels[mask] if mask is not None else pixels.reshape(-1, pixels.shape[-1])
    return classify_colour(np.median(pixels[:, :3], axis=0))


# Keep file loading separate from classification so callers can supply an image array too.
def classify_image_file(path):
    """Load an image file and classify its dominant colour."""
    from PIL import Image

    with Image.open(path) as img:
        return classify_image(np.asarray(img.convert("RGB")))


# A small image-file stand-in lets the simulated cell exercise the camera workflow without hardware.
class SimulatedCamera:
    """Stand-in for a real camera: when the object flag is raised, 'captures' one of the image files at random."""

    # Save paths and segmentation thresholds once; the camera stays idle until the part-present flag is set.
    def __init__(
        self,
        folder=None,
        filenames=("black.png", "brown.png"),
        white_saturation_threshold=WHITE_BACKGROUND_MAX_SATURATION,
        dark_pixel_threshold=DARK_FOREGROUND_MAX_VALUE,
    ):
        """Set the image source and thresholds for excluding the white background."""
        folder = folder or os.path.dirname(os.path.abspath(__file__))
        self.image_paths = [os.path.join(folder, name) for name in filenames]
        self.white_saturation_threshold = white_saturation_threshold
        self.dark_pixel_threshold = dark_pixel_threshold
        self.object_present = False
        self._fig = None
        self._closed_figs = []

    # Random choice makes repeated demos look like separate camera captures.
    def capture(self):
        """Return the path of a randomly selected image, as if photographing the object."""
        return random.choice(self.image_paths)

    # Ignore the white backdrop, keep its largest connected patch, and classify only those part pixels.
    def detect_colour(self, image_path=None):
        """If the object flag is set, capture an image, show it, and return its classified colour, else None."""
        if not self.object_present:
            return None
        from PIL import Image

        path = self.capture() if image_path is None else os.fspath(image_path)
        with Image.open(path) as img:
            pixels = np.asarray(img.convert("RGB"), dtype=float)
        # Saturation separates coloured parts from white; the max-channel test catches very dark parts.
        max_channel = pixels.max(axis=2)
        min_channel = pixels.min(axis=2)
        saturation = np.divide(
            max_channel - min_channel,
            max_channel,
            out=np.zeros_like(max_channel),
            where=max_channel > 0,
        )
        foreground = (saturation >= self.white_saturation_threshold) | (
            max_channel <= self.dark_pixel_threshold
        )
        # Connected components avoid treating stray noise pixels as part of the object.
        labels, _ = ndimage.label(foreground)
        component_sizes = np.bincount(labels.ravel())
        component_sizes[0] = 0
        largest_component = int(component_sizes.argmax())
        foreground = labels == largest_component
        if not foreground.any():
            raise ValueError(
                f"No part foreground found in {path}; check the white-background "
                "segmentation thresholds."
            )
        colour_pixels = pixels[foreground]
        colour = classify_image(pixels, mask=foreground)
        # Keep the measured colour in 0-1 form for anyone inspecting the last simulated capture.
        self.last_rgb = tuple(
            float(c) for c in np.median(colour_pixels, axis=0) / 255.0
        )
        print(f"Camera image {os.path.basename(path)} -> {colour}")
        # Show only the part: crop to the mask's bounding box (plus padding) so the white background isn't displayed
        rows, cols = np.where(foreground)
        pad = DISPLAY_PADDING_PX
        top, bottom = max(rows.min() - pad, 0), min(rows.max() + pad + 1, pixels.shape[0])
        left, right = max(cols.min() - pad, 0), min(cols.max() + pad + 1, pixels.shape[1])
        cropped = pixels[top:bottom, left:right].astype(np.uint8)
        self.show_image(cropped, f"Camera: {os.path.basename(path)} -> {colour}")
        return colour

    # Replace the old preview rather than leaving a pile of windows open during the run.
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

    # Called when the cell closes the camera preview.
    def close_image(self):
        """Close the image window if it is open."""
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            # Tk objects must not be garbage-collected from another thread (the Do robot's),
            # so keep the closed figure alive instead of letting it be freed there
            self._closed_figs.append(self._fig)
            self._fig = None

    # Let the GUI event loop repaint the preview while the simulation keeps stepping.
    def pump(self):
        """Keep the image window responsive while the simulation runs."""
        if self._fig is not None:
            self._fig.canvas.flush_events()


# Quick manual checks for the colour thresholds when running this file directly.
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
