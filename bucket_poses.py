from spatialmath import SE3

# World-frame part and drop poses in metres; adjust to match the scene.
GENERAL_BUCKET_POSE = SE3(0.55, 0.0, 0.10)
GREEN_BUCKET_POSE = SE3(0.25, 0.40, 0.10)
RED_BUCKET_POSE = SE3(0.25, 0.0, 0.10)
BLUE_BUCKET_POSE = SE3(0.25, -0.40, 0.10)
BLACK_BUCKET_POSE = SE3(0.25, 0.20, 0.10)
BROWN_BUCKET_POSE = SE3(0.25, -0.20, 0.10)
IMAGE_COLOUR_BUCKET_POSES = {
    "black": BLACK_BUCKET_POSE,
    "brown": BROWN_BUCKET_POSE,
}
OBJ_COLOUR = "red"
COLOUR_BUCKET_POSES = {
    "green": GREEN_BUCKET_POSE,
    "red": RED_BUCKET_POSE,
    "blue": BLUE_BUCKET_POSE,
}
