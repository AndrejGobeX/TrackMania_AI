from time import sleep
from TMEnv import TMEnv
from stable_baselines3 import SAC
import sys
import numpy as np

print("Init")
np.set_printoptions(suppress=True)
env = TMEnv(sys.argv[1])
model = SAC('MlpPolicy', env, verbose=1)

def train(i):
  global model
  while i:
    i-=1
    model.learn(total_timesteps=3000, reset_num_timesteps=False)
    model.save('./RL_Models/sac.zip')

#model = SAC.load('./RL_Models/SACv3.zip', verbose=1, learning_rate=0.0003)
#model.set_env(env)

#train(100)

print("Eval")

obs = env.reset()
episode_reward = 0.0
for i in range(5000):
  #action, _state = np.array([0,0,0]), 0
  action, _state = model.predict(obs, deterministic=True)
  obs, reward, done, info = env.step(action)
  episode_reward += reward
  #print(reward)
  if done:
    print(episode_reward)
    episode_reward = 0.0
    obs = env.reset()