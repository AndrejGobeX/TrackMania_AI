import os

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from TrackmaniaEnv import TrackmaniaEnv as TMEnv

from stable_baselines3.sac import SAC
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3

# config
ALG = 'TD3'
comment = '_equalizer_512_0o_2a'
path = 'Algs\\'+ALG+comment


# callbacks
class LaptimeCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        global env
        if not env.lap_time is None:
            self.logger.record("eval/lap_time", env.lap_time)
            print(env.lap_time)
        return True


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
        return True
    

policy = 'MlpPolicy'
run_name = "second_run"

log_path = os.path.join(path, "logs")
tensorboard_path = os.path.join(path, "tensorboard")
replay_buffer_path = os.path.join(path, "replay_buffer.pkl")
best_model_path = os.path.join(log_path, "best_model.zip")
map_file = '.\\Maps\\Sunset.Map.txt'
#load_replay = False
reset_timesteps = False

# envs
env = Monitor(TMEnv(map_file, human_driver=False))

# callbacks
checkpoint_callback = CheckpointCallback(save_freq=2_000, save_path=log_path, name_prefix=ALG)
lap_time_callback = LaptimeCallback()
eval_callback = EvalCallback(eval_env=env, best_model_save_path=log_path, eval_freq=2_000, callback_after_eval=lap_time_callback)
save_replay_buffer_callback = SaveReplayBufferCallback(save_path=replay_buffer_path, save_freq=2_000)
callbacks = [checkpoint_callback, eval_callback]#, save_replay_buffer_callback, ]

# training
if os.path.exists(best_model_path):
    model = TD3.load(best_model_path, env=env, verbose=2, tensorboard_log=tensorboard_path)
else:
    model = TD3(policy, env=env, verbose=2, gamma=0.999, tensorboard_log=tensorboard_path, device='cpu')

if os.path.exists(replay_buffer_path):
    model.load_replay_buffer(replay_buffer_path)

model.learn(total_timesteps=100_000, tb_log_name=run_name, reset_num_timesteps=reset_timesteps, callback=callbacks)