import gym
from gym import spaces

import numpy as np
import cv2
import math
import time

from MapExtractor import get_map_data

from Commands import (
  tm_accelerate,
  tm_brake,
  tm_steer,
  tm_reset,
  tm_respawn,
  tm_update
)

from GameDataGetter import GameDataGetter
from Window import WindowInterface
from Lidar import Lidar


class TMImageEnv(gym.Env):


  def __init__(self, map_path:str, obs_history:int=0, action_history:int=2, human_driver=False):
    """Gym(nasium) compatible env for imitation/reinforcement learning in Trackmania.

    ### Parameters
    map_path : str
        Path to processed track file.
    obs_history : int, (default 1)
        Number of previous inputs to keep in the observation.
    action_history : int, (default 2)
        Number of previous actions to keep in the observation.
        The total number of previous actions will be equal to:
        obs_history + action_history
    human_driver : bool, (default False)
        If True, env will not apply any action.
    """

    super().__init__()

    # *** ACTION SPACE ***

    self.action_space = spaces.Box(
      # steer, gas/brake
      np.array([-1, -1], dtype=np.float32),
      np.array([1, 1], dtype=np.float32)
    )

    self.obs_history = obs_history
    self.action_history = action_history

    # *** OBSERVATION SPACE ***
    
    self.observation_space = spaces.Dict({})

    for i in range(self.obs_history+1):
      self.observation_space['view_'+str(i)] = spaces.Box(
        low=0, high=255, shape=(100,100,3), dtype=np.uint8
      )
    
    self.observation_space['speed'] = spaces.Box(
      np.array([0]*(self.obs_history+1)),
      np.array([1]*(self.obs_history+1)),

      dtype=np.float32
    )

    self.observation_space['prev'] = spaces.Box(
      np.array([[-1,-1]]*(self.obs_history+self.action_history)),
      np.array([[1,1]]*(self.obs_history+self.action_history)),

      dtype=np.float32
    )

    # *** SETUP ***

    # flipping mechanism
    self.flip = False

    # respawning (true first time)
    self.rspwn = True

    # human driver mode
    self.human = human_driver

    # load map
    self.map = get_map_data(map_path)
    self.map_centerline = self.map.reshape((-1,2,2)).sum(axis=1)/2 # x, y
    self.next_checkpoint = 1
    self.location = np.array([0,0], dtype=np.float32)
    self.prev_location = np.array([0,0], dtype=np.float32)
    self.prev_projection = np.array([0,0], dtype=np.float32)
    self.direction = np.array([0,0], dtype=np.float32)
    self.prev_distance = 0.0

    # setup Trackmania bridge
    self.data_getter = GameDataGetter(extended=True)

    # setup window capture
    self.window = WindowInterface()
    self.window.move_and_resize()
    self.lidar = Lidar(self.window.screenshot())

    # obs buffer
    self.view_buffer = np.zeros((self.obs_history+1, 100, 100, 3), dtype=np.uint8)
    self.speed_buffer = np.zeros((self.obs_history+1), dtype=np.float32)
    self.prev_action_buffer = np.zeros((self.obs_history+self.action_history, 2), np.float32)

    self.obs_dict = dict({})
    for i in range(self.obs_history+1):
      self.obs_dict['view_'+str(i)] = self.view_buffer[i]
    self.obs_dict['speed'] = self.speed_buffer
    self.obs_dict['prev'] = self.prev_action_buffer

  # *** GAME MANIP ***

  def respawn(self):
    """Respawns the car to the start.
    """
    tm_reset()
    tm_update()
    tm_respawn()
    time.sleep(0.4)
    print("respawned")
    self.next_checkpoint = 1
    self.location[0] = self.data_getter.game_data[GameDataGetter.I_X]
    self.location[1] = self.data_getter.game_data[GameDataGetter.I_Z]
    self.prev_location[0] = self.location[0]
    self.prev_location[1] = self.location[1]
    self.prev_projection = self.location.copy()
    self.prev_distance = 0.0
    self.flip = not self.flip
    for i in range(self.obs_history+self.action_history+1):
      self.refresh_observation()
    print("go")
    #time.sleep(2)


  def refresh_observation(self):
    """Obtains the observation and updates the buffer.
    """
    # overwrite
    self.view_buffer[:-1] = self.view_buffer[1:]
    self.speed_buffer[:-1] = self.speed_buffer[1:]
    
    # add new
    view = cv2.cvtColor(self.window.screenshot(), cv2.COLOR_RGBA2RGB)
    view = cv2.resize(view, (100,100))
    if self.flip:
      view = np.flip(view, axis=1)
    self.view_buffer[-1] = view

    speed = self.data_getter.game_data[
      GameDataGetter.I_SPEED
    ]*0.0036
    self.speed_buffer[-1] = speed


  def apply_action(self, action:np.ndarray):
    """Applies the given action.

    ### Parameters
    action : np.ndarray
        Array of two floats [-1,1] indicating, respectively, steering and throttle/braking.
    """
    # overwrite
    self.prev_action_buffer[:-1] = self.prev_action_buffer[1:]
    
    # add new
    self.prev_action_buffer[-1:] = action

    if self.flip:
      action[0] = -action[0]
    
    if not self.human:
      tm_reset()

      tm_steer(action[0])
      if action[1] > 0.0:
        tm_accelerate(action[1])
      elif self.speed_buffer[-1] > 0.07:
        tm_brake(-action[1])
      
      tm_update()

  # *** UTILITIES ***

  def norm(x):
    return math.sqrt(x[0]**2 + x[1]**2)

  
  def cross_product(x, y):
    return x[0]*y[1] - x[1]*y[0]


  def vector_angle(r):
    n = TMImageEnv.norm(r)
    if n == 0.0:
      return 0.0
    v = r[0]/n
    θ = np.arccos(v)
    if r[1] < 0:
      θ *= -1
    return θ


  def vector_intersection(p, r, q, s):
    rxs = TMImageEnv.cross_product(r, s)
    qmp = q - p
    qpxs = TMImageEnv.cross_product(qmp, s)
    qpxr = TMImageEnv.cross_product(qmp, r)
    if rxs == 0:
      return None
    t = qpxs/rxs
    u = qpxr/rxs
    if t >= 0 and t <= 1 and u >= 0 and u <= 1:
      return p + t*r
    return None

  # *** ENV ESSENTIALS ***
  
  def calc_reward(self):
    # calc distance travelled between two steps
    centerline_distance = 0.0

    self.prev_location[0] = self.location[0]
    self.prev_location[1] = self.location[1]
    self.location[0] = self.data_getter.game_data[GameDataGetter.I_X]
    self.location[1] = self.data_getter.game_data[GameDataGetter.I_Z]
    self.direction[0] = self.data_getter.game_data[GameDataGetter.I_DX]
    self.direction[1] = self.data_getter.game_data[GameDataGetter.I_DZ]
    
    v = -self.direction*100

    d_angle = abs(TMImageEnv.vector_angle(self.map_centerline[self.next_checkpoint] - self.map_centerline[self.next_checkpoint-1]) - \
      TMImageEnv.vector_angle(-v))
    
    if d_angle > math.pi:
      d_angle = 2*math.pi - d_angle

    if d_angle > math.pi/2:
      self.rspwn = True
      return -100

    w = self.map[self.next_checkpoint][0:2] - self.map[self.next_checkpoint][2:4]
    while not TMImageEnv.vector_intersection(
      self.location,
      v,
      self.map[self.next_checkpoint][2:4],
      w
      ) is None:
            centerline_distance += TMImageEnv.norm(self.map_centerline[self.next_checkpoint] - self.prev_projection)
            self.prev_projection = self.map_centerline[self.next_checkpoint]
            self.next_checkpoint += 1
            if self.next_checkpoint >= len(self.map_centerline):
              self.next_checkpoint = 1
              self.rspwn = True
            w = self.map[self.next_checkpoint][0:2] - self.map[self.next_checkpoint][2:4]

    # projection on the centerline
    k = self.map_centerline[self.next_checkpoint][0:2]-self.map_centerline[self.next_checkpoint-1][0:2]
    if k[0] == 0:
        projection = np.array([self.map_centerline[self.next_checkpoint][0],self.location[1]])
    else:
        k = k[1]/k[0]
        n = self.map_centerline[self.next_checkpoint][1] - k*self.map_centerline[self.next_checkpoint][0]
        projection = (self.location[0] + k*self.location[1] - k*n)/(1+k**2)
        projection = np.array([
            projection,
            projection*k + n
        ])
    
    n1 = TMImageEnv.norm(self.map_centerline[self.next_checkpoint]-self.prev_projection)
    n2 = TMImageEnv.norm(self.map_centerline[self.next_checkpoint]-projection)
    # advancement
    if n2 < n1:
      centerline_distance += n1-n2
      self.prev_projection = projection

    return centerline_distance


  def reset(self):
    self.prev_time = time.time()
    if self.rspwn or self.data_getter.game_data[
      GameDataGetter.I_SPEED
    ] * 3.6 < 2.0 or self.data_getter.game_data[
      GameDataGetter.I_FINISH
    ]:
      self.rspwn = False
      self.respawn()

    return self.obs_dict


  def step(self, action):
    print(time.time() - self.prev_time)
    self.prev_time = time.time()
    self.apply_action(action)
    self.refresh_observation()
    reward = self.calc_reward()

    if self.rspwn or self.data_getter.game_data[
      GameDataGetter.I_FINISH
    ]:
      self.rspwn = False
      self.respawn()
    
    return self.obs_dict, reward, False, {}
