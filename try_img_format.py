from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
import cv2
from metadrive.policy.idm_policy import IDMPolicy
from IPython.display import Image
import numpy as np
import os
from PIL import Image as PILImage

sensor_size = (84, 60) if os.getenv('TEST_DOC') else (200, 100)

cfg = dict(
    image_observation=True,
    vehicle_config=dict(image_source="rgb_camera"),
    sensors={"rgb_camera": (RGBCamera, *sensor_size)},
    stack_size=3,
    agent_policy=IDMPolicy,  # drive with IDM policy
    use_render=False
)

env = MetaDriveEnv(cfg)
frames = []
try:
    env.reset()
    for _ in range(1 if os.getenv('TEST_DOC') else 10000):
        # simulation
        o, r, d, _, _ = env.step([0, 1])
        # rendering, the last one is the current frame
        ret = o["image"][..., -1] * 255  # [0., 1.] to [0, 255]
        ret = ret.astype(np.uint8)
        frames.append(ret[..., ::-1])  # Convert from BGR to RGB
        if d:
            break

    # Convert one of the frames to a PIL Image and save it for inspection
    if len(frames) > 0:
        pil_image = PILImage.fromarray(frames[-1])  # Use the last frame for inspection
        pil_image.save("test_image.png")
        # Optionally display the image in the notebook - you can remove this part if not needed
finally:
    env.close()
