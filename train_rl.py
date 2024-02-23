import argparse
import os
import os.path as osp
import torch

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.idm_policy import IDMPolicy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from utils import get_time_str
from wandb_callback import WandbCallback
from vlm_extractor import CustomVLMFeatureExtractor
baseline_eval_config = dict(manual_control=False, use_render=False, start_seed=1000, horizon=1500)

def make_eval_env(log_dir):
    def _init():
        env = Monitor(env=MetaDriveEnv(config=baseline_eval_config), filename=os.path.join(log_dir, "eval"))
        return env

    return _init


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="vlm_rl_initial_test", type=str, help="The experiment name.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--device", default="cuda:2", type=str, help="The CUDA Device")
    parser.add_argument("--feature-dim", default=128, type=int, help="The feature dim used for RL")
    parser.add_argument("--num-transformer-layer", default=4, type=int, help="The #layer of transformer")
    args = parser.parse_args()

    # ===== Setup some meta information =====
    exp_name = args.exp_name
    seed = int(args.seed)
    use_wandb = args.wandb
    device = torch.device(args.device)
    feature_dim = args.feature_dim
    num_transformer_layer = args.num_transformer_layer
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    project_name = "vqa_rl"
    team_name = "chenda_playground"

    trial_name = "{}_seed{}_{}".format(exp_name, seed, get_time_str())
    log_dir = osp.join("runs", exp_name, trial_name)
    os.makedirs(osp.join("runs", exp_name), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print("We start logging training data into {}".format(log_dir))

    # ===== Setup the config =====
    config = dict(
        # Environment config
        env_config = dict(
            use_render=False,
            start_seed=0,
            num_scenarios=400,
            image_on_cuda=False,
            traffic_density=0.2,
            image_observation=True,
            sensors={
                "rgb_camera": (RGBCamera, 256, 256),
            },
            agent_policy=IDMPolicy,
            interface_panel=[],
            vehicle_config={
                "image_source": "rgb_camera",
            },
        ),

        # Algorithm config
        algo=dict(
            policy="MlpPolicy",
            policy_kwargs=dict(net_arch=[128],
                               features_extractor_class=CustomVLMFeatureExtractor,
                               features_extractor_kwargs=dict(features_dim=feature_dim,
                                                              num_transformer_layer=num_transformer_layer,
                                                              device = device),
                               ),
            env=None,
            # TODO: Change learning rate
            learning_rate=1e-4,

            batch_size=64,  # Reduce the batch size for real-time copilot
            gamma=0.99,
            tensorboard_log=log_dir,
            verbose=2,
            seed=seed,
            device = device,
        ),

        # Meta data
        project_name=project_name,
        team_name=team_name,
        exp_name=exp_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=log_dir
    )

    # Need this for rendering on server
    env = MetaDriveEnv({"use_render": False, "image_observation": False})
    try:
        env.reset()
        for i in range(1, 100):
            o, r, tm, tc, info = env.step([0, 1])
    finally:
        env.close()

    # ===== Setup the training environment =====k
    train_env = MetaDriveEnv(config=config["env_config"], )
    eval_env = SubprocVecEnv([make_eval_env(log_dir)])
    train_env = Monitor(env=train_env, filename=log_dir)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Setup the callbacks =====
    callbacks = [
        CheckpointCallback(
            name_prefix="rl_model",
            verbose=1,
            save_freq=10000,
            save_path=osp.join(log_dir, "models")
        )
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=exp_name,
                project_name=project_name,
                team_name=team_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    model = PPO(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=200_0000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        # eval_env=eval_env,
        # eval_freq=5000,
        # n_eval_episodes=30,
        # eval_log_path=log_dir,

        # logging
        tb_log_name=exp_name,  # Should place the algorithm name here!
        log_interval=4,
    )