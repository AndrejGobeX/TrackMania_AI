import numpy as np
import time
from gym.wrappers import TimeLimit

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

import keyboard

from TrackmaniaEnv import TrackmaniaEnv


# *** META ***
MAX_EPISODE_STEPS = 500
BUFFER_SIZE = 20000
NUMBER_EPISODES = 10

# *** INIT ***
env = TrackmaniaEnv('./Maps/TurboTrack.Map.txt', human_driver=True)
#check_env(env)
envT = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
model = SAC("MlpPolicy", envT, buffer_size=BUFFER_SIZE, verbose=1)

# # *** BNCHMRK ***
# obs = envT.reset()
# done = False
# total = 0.0
# cnt = 0
# while not done:
#     start = time.time()
#     obs, reward, done, info = envT.step(model.predict(obs)[0])
#     total += time.time() - start
#     cnt += 1
# print(total/cnt)
# exit()

# *** RUN ***
for i in range(NUMBER_EPISODES):
    observations = []
    actions = []
    rewards = []
    infos = []
    act = np.array([0,0], dtype=np.float64)
    obs = envT.reset()
    observations.append(obs)
    done = False
    crash = 'OK'
    print(obs)

    while not done:

        gas = keyboard.is_pressed('up') - \
        keyboard.is_pressed('down')

        steer = keyboard.is_pressed('right') - \
        keyboard.is_pressed('left')

        if keyboard.is_pressed('f'):
            crash = 'FAIL'

        actions.append([
        steer * (-1 if env.flip else 1),
        gas
        ])

        obs, reward, done, info = envT.step(actions[-1])

        observations.append(obs)
        print(reward)
        rewards.append(reward)
        
        infos.append(info)
        #print(actions[-1], reward)
        #print(env.next_checkpoint)

    np.savez('Trajectories/trajectory_' + str(int(time.time())) + '_' + crash, obs=observations, act=actions, inf=infos, rews=rewards)
