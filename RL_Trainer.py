# RL StableBaselines3 trainer
# python RL_Trainer.py path_to_processed_map run_name --no-replay --reset
import os
from termcolor import colored
import numpy as np
import sys
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from gym.wrappers.time_limit import TimeLimit
from TrackmaniaEnv import TrackmaniaEnv as TMEnv

ALG = 'PPO'

if ALG == 'SAC':
    from stable_baselines3 import SAC as Algo
if ALG == 'PPO':
    from stable_baselines3 import PPO as Algo


class SaveReplayBufferCallback(BaseCallback):
    def __init__(self, save_path, save_freq=1000, verbose=0):
        super(SaveReplayBufferCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.counter = 0

    def _on_step(self):
        self.counter += 1
        if self.counter == self.save_freq:
            self.counter = 0
            print("Saving replay buffer to " + self.save_path)
            self.model.save_replay_buffer(self.save_path)
            print(colored("Evaluation", "blue"))
        return True


# paths
path = 'Algs'
policy = 'MlpPolicy'

log_path = os.path.join(path, "logs")
tensorboard_path = os.path.join(path, "tensorboard/")
replay_buffer_path = os.path.join(path, "replay_buffer.pkl")
best_model_path = os.path.join(log_path, "best_model.zip")
map_file = '.\\Maps\\TurboTrack.Map.txt'
run_name = "first_run"
load_replay = True
reset_timesteps = False

# parameters
if len(sys.argv) >= 2:
    map_file = sys.argv[1]
if len(sys.argv) >= 3:
    run_name = sys.argv[2]
if len(sys.argv) >= 4:
    load_replay = False
if len(sys.argv) >= 5:
    reset_timesteps = True

# envs
env = Monitor(TMEnv(map_file))

# callbacks
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_path, name_prefix=ALG)
eval_callback = EvalCallback(eval_env=env, best_model_save_path=log_path, eval_freq=1000)
save_replay_buffer_callback = SaveReplayBufferCallback(save_path=replay_buffer_path, save_freq=1000)
callbacks = [save_replay_buffer_callback, checkpoint_callback, eval_callback]
if ALG in ['PPO']:
    callbacks = [checkpoint_callback, eval_callback]

# learning rate
initial_learning_rate = 0.003
final_learning_rate = 0.0003
d_lr = initial_learning_rate - final_learning_rate

def linear_schedule(progress_remaining):
        return progress_remaining * d_lr + final_learning_rate

# training
if os.path.exists(best_model_path):
    model = Algo.load(best_model_path, env=env, verbose=2, learning_rate=initial_learning_rate, tensorboard_log=tensorboard_path)
else:
    model = Algo(policy, env=env, verbose=2, learning_rate=initial_learning_rate, tensorboard_log=tensorboard_path)

if os.path.exists(replay_buffer_path) and load_replay:
    model.load_replay_buffer(replay_buffer_path)

model.learn(total_timesteps=100000, tb_log_name=run_name, reset_num_timesteps=reset_timesteps, callback=callbacks)

# obs = env.reset()
# episode_reward = 0.0
# for i in range(5000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     episode_reward += reward
#     print(episode_reward, '\t\t', reward)
#     if done:
#         print("Episode reward:", episode_reward)
#         episode_reward = 0.0
#         obs = env.reset()