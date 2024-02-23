from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
import cv2
from metadrive.policy.idm_policy import IDMPolicy
from IPython.display import Image
import numpy as np
import os
import sys
from PIL import Image as PILImage
from metadrive.utils import generate_gif

sensor_size = (84, 60) if os.getenv('TEST_DOC') else (200, 100)

env = MetaDriveEnv({"use_render": False, "image_observation": False})
try:
    env.reset()
    for i in range(1, 100):
        o, r, tm, tc, info = env.step([0, 1])
finally:
    frames = []
    env.close()
    try:
        env = MetaDriveEnv(
            dict(
                use_render=False,
                start_seed=666,
                image_on_cuda=False,
                traffic_density=0.1,
                image_observation=True,
                window_size=(600, 400),
                sensors={
                    "rgb_camera": (RGBCamera, 256, 256),
                },
                interface_panel=[],
                vehicle_config={
                    "image_source": "rgb_camera",
                },
            )
        )
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
            test = frames[-1]
            # print(type(test))
            # test_numpy = test.get()
            pil_image = PILImage.fromarray(test)  # Use the last frame for inspection
            pil_image.save("test_image.png")
            # Optionally display the image in the notebook - you can remove this part if not needed
        generate_gif(frames if os.getenv('TEST_DOC') else frames[-300:-50])
    finally:
        env.close()
