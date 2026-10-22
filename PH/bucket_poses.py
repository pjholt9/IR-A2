from spatialmath import SE3

# I keep every pick/drop target in world metres so the sorter can use one pose convention.
# World-frame part and drop poses in metres; adjust to match the scene.
GENERAL_BUCKET_POSE = SE3(0.55, 0.0, 0.10)
GREEN_BUCKET_POSE = SE3(0.25, 0.40, 0.10)
RED_BUCKET_POSE = SE3(0.25, 0.0, 0.10)
BLUE_BUCKET_POSE = SE3(0.25, -0.40, 0.10)
BLACK_BUCKET_POSE = SE3(0.25, 0.20, 0.10)
BROWN_BUCKET_POSE = SE3(0.25, -0.20, 0.10)
# Image-only colours have their own drop locations, separate from the RGB buckets.
IMAGE_COLOUR_BUCKET_POSES = {
    "black": BLACK_BUCKET_POSE,
    "brown": BROWN_BUCKET_POSE,
}
# This is the example part colour used by the sorting setup.
OBJ_COLOUR = "red"
# The camera classification selects one of these destination poses.
COLOUR_BUCKET_POSES = {
    "green": GREEN_BUCKET_POSE,
    "red": RED_BUCKET_POSE,
    "blue": BLUE_BUCKET_POSE,
}
