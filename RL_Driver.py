from time import sleep, time
from TMEnv import TMEnv
from stable_baselines3 import SAC
import sys
import numpy as np

print("Init")
np.set_printoptions(suppress=True)
env = TMEnv(sys.argv[1])
model = SAC('MlpPolicy', env, verbose=2)

def train(i):
  global model
  while i:
    i-=1
    model.learn(total_timesteps=3000, reset_num_timesteps=False)
    model.save('./RL_Models/sac_neo.zip')

#model = SAC.load('./RL_Models/sac_neo.zip', verbose=2, learning_rate=0.0003)
#model.set_env(env)

train(1000)

print("Eval")

obs = env.reset()
episode_reward = 0.0
action, _state = np.array([0,0]), 0
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