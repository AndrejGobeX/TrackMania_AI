import os

from TrackmaniaEnv import TrackmaniaEnv as TMEnv

from stable_baselines3.sac import SAC
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3

# config
ALG = 'PPO'
comment = '_equalizer_512_0o_2a'
path = 'Algs\\'+ALG+comment

log_path = os.path.join(path, "logs")
tensorboard_path = os.path.join(path, "tensorboard")
replay_buffer_path = os.path.join(path, "replay_buffer.pkl")
best_model_path = os.path.join(log_path, "best_model.zip")
map_file = '.\\Maps\\Test2.Map.txt'
#load_replay = False
reset_timesteps = False

# envs
env = TMEnv(map_file, start_delay=0.0, human_driver=False)

# training
model = PPO.load(best_model_path, env=env, verbose=2, tensorboard_log=tensorboard_path)

for i in range(10):
    obs, _ = env.reset()
    done = False
    while not done:
        obs, rew, done, _, _ = env.step( model.predict(obs, deterministic=True)[0] )
    print(env.lap_time)