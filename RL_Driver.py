import os
from time import sleep, time
from TMEnv import TMEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import sys
import numpy as np

def drive():
  obs = env.reset()
  episode_reward = 0.0
  action, _state = np.array([0,0]), 0
  for i in range(5000):
    start = time()
    obs, reward, done, info = env.step(action)
    #print(time()-start)
    episode_reward += reward
    print(episode_reward, '\t\t', reward)
    if done:
      print("Episode reward:", episode_reward)
      episode_reward = 0.0
      obs = env.reset()

model_path = './RL_Models/sac_neo.zip'
replay_path = './RL_Models/sac_neo_rb'
log_path = './RL_Models/sac_neo_tensorboard/'

np.set_printoptions(suppress=True)
env = TMEnv(sys.argv[1])

model = ''

if len(sys.argv) > 2:
  if sys.argv[2] == '--test':
    print(env.observation_space.sample())
    check_env(env)
    exit()
  if sys.argv[2] == '--freeroam':
    drive()
    exit()
  if sys.argv[2] == '--load':
    model = SAC.load(model_path, verbose=2, tensorboard_log=log_path)
    model.set_env(env)
    model.load_replay_buffer(replay_path)
if model == '':
  model = SAC('MlpPolicy', env, verbose=2, tensorboard_log=log_path)
  #os.unlink(log_path)


def train(i):
  global model
  while i:
    i-=1
    model.learn(total_timesteps=1000, reset_num_timesteps=False)
    model.save(model_path)
    model.save_replay_buffer(replay_path)


train(1000)

print("Finished training")

obs = env.reset()
episode_reward = 0.0
for i in range(5000):
  action, _state = model.predict(obs, deterministic=True)
  start = time()
  obs, reward, done, info = env.step(action)
  print(time()-start)
  episode_reward += reward
  #print(episode_reward, '\t\t', reward)
  if done:
    print("Episode reward:", episode_reward)
    episode_reward = 0.0
    obs = env.reset()