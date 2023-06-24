from gym import Env, spaces

import numpy as np
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
from TrackVisualizer import TrackVisualizer


class TrackmaniaEnv(Env):


  def __init__(self, map_path:str, obs_history:int=0, action_history:int=2, human_driver:bool=False):
    """Gym(nasium) compatible env for imitation/reinforcement learning in Trackmania.

    ### Parameters
    map_path : str
        Path to processed track file.
    obs_history : int, (default 0)
        Number of previous inputs to keep in the observation.
    action_history : int, (default 2)
        Number of previous actions to keep in the observation.
        The total number of previous actions will be equal to:
        obs_history + action_history
    max_steps: int, (default 200)
        Maximum number of timesteps before restart.
    human_driver : bool, (default False)
        If True, env will not apply any action.
    """

    super().__init__()
    wall_lasers=13

    # *** ACTION SPACE ***

    self.action_space = spaces.Box(
      # steer, gas/brake
      np.array([-1, -1], dtype=np.float32),
      np.array([1, 1], dtype=np.float32)
    )

    self.obs_history = obs_history
    self.action_history = action_history

    # *** OBSERVATION SPACE ***
    
    self.observation_space = spaces.Box(
      np.array(
        [-1, -1]*(self.obs_history+self.action_history) + \
        [0]*(wall_lasers+1)*(self.obs_history+1),
        
        dtype=np.float32
      ),
      np.array(
        [1, 1]*(self.obs_history+self.action_history) + \
        [1]*(wall_lasers+1)*(self.obs_history+1),
        
        dtype=np.float32
      )
    )

    # *** SETUP ***

    # flipping mechanism
    self.flip = True

    # respawning (true first time)
    self.rspwn = True
    self.done = False
    #self.threshold_speed = False

    # human driver mode
    self.human = human_driver

    # load map
    self.map = get_map_data(map_path)
    self.map_centerline = self.map.reshape((-1,2,2)).sum(axis=1)/2 # x, y
    self.next_checkpoint = 1
    self.location = np.array([0,0], dtype=np.float32)
    self.prev_projection = np.array([0,0], dtype=np.float32)
    self.direction = np.array([0,0], dtype=np.float32)
    self.prev_distance = 0.0

    # setup Trackmania bridge
    self.data_getter = GameDataGetter(extended=True)

    # setup visualizer and lidar
    self.visualizer = TrackVisualizer(self.map)
    self.wall_number = wall_lasers

    # obs buffer
    self.obs_buffer = np.array(
      [0, 0]*(self.obs_history+self.action_history) + \
      [0]*(wall_lasers+1)*(self.obs_history+1),

      dtype=np.float32
    )
    self.view_buffer = self.obs_buffer[2*(self.obs_history+self.action_history) + self.obs_history + 1:]
    self.speed_buffer = self.obs_buffer[2*(self.obs_history+self.action_history):
      2*(self.obs_history+self.action_history) + self.obs_history + 1]
    self.prev_action_buffer = self.obs_buffer[:2*(self.obs_history+self.action_history)]

  # *** GAME MANIP ***

  def respawn(self):
    """Respawns the car to the start.
    """
    tm_reset()
    tm_update()
    tm_respawn()
    time.sleep(1.5)

    self.next_checkpoint = 1
    self.location[0] = self.data_getter.game_data[GameDataGetter.I_X]
    self.location[1] = self.data_getter.game_data[GameDataGetter.I_Z]
    self.prev_projection = self.location.copy()
    self.prev_distance = 0.0
    #self.threshold_speed = False
    self.flip = not self.flip
    for i in range(self.obs_history+self.action_history):
      self.refresh_observation()


  def refresh_observation(self):
    """Obtains the observation and updates the buffer.
    """
    # get info from the game
    self.location[0] = self.data_getter.game_data[GameDataGetter.I_X]
    self.location[1] = self.data_getter.game_data[GameDataGetter.I_Z]
    self.direction[0] = self.data_getter.game_data[GameDataGetter.I_DX]
    self.direction[1] = self.data_getter.game_data[GameDataGetter.I_DZ]

    # overwrite
    self.view_buffer[:-self.wall_number] = self.view_buffer[self.wall_number:]
    self.speed_buffer[:-1] = self.speed_buffer[1:]
    
    # add new
    view = self.visualizer.lidar(self.location, TrackmaniaEnv.vector_angle(self.direction), show=False)
    if self.flip:
      view = np.flip(view)
    self.view_buffer[-self.wall_number:] = view

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
    self.prev_action_buffer[:-2] = self.prev_action_buffer[2:]
    
    # add new
    self.prev_action_buffer[-2:] = action

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
    n = TrackmaniaEnv.norm(r)
    if n == 0.0:
      return 0.0
    v = r[0]/n
    θ = math.acos(v)
    if r[1] < 0:
      θ *= -1
    return θ


  def vector_intersection(p, r, q, s):
    rxs = TrackmaniaEnv.cross_product(r, s)
    qmp = q - p
    qpxs = TrackmaniaEnv.cross_product(qmp, s)
    qpxr = TrackmaniaEnv.cross_product(qmp, r)
    if rxs == 0:
      return None
    t = qpxs/rxs
    u = qpxr/rxs
    if t >= 0 and t <= 1 and u >= 0 and u <= 1:
      return p + t*r
    return None

  
  def normal_projection(a, b, p):
    v = a - b
    v /= TrackmaniaEnv.norm(v)
    w = p - b
    return b + v*(w[0]*v[0] + w[1]*v[1])
  
  # *** ENV ESSENTIALS ***
  
  def calc_reward(self):
    # calc distance travelled between two steps
    centerline_distance = 0.0
    
    # negative direction vector
    v = -self.direction*100
    
    car_angle = TrackmaniaEnv.vector_angle(-v)
    centerline_angle = TrackmaniaEnv.vector_angle(self.map_centerline[self.next_checkpoint] - self.map_centerline[self.next_checkpoint-1])

    # delta angle between centerline and direction
    d_angle = car_angle - centerline_angle

    if d_angle > math.pi: d_angle -= 2*math.pi
    elif d_angle < -math.pi: d_angle += 2*math.pi

    if abs(d_angle) > math.pi/2:
      self.rspwn = True
      self.done = True
      return -500.0
    
    if self.flip:
      d_angle = - d_angle

    # checkpoints
    w = self.map[self.next_checkpoint][0:2] - self.map[self.next_checkpoint][2:4]
    while not TrackmaniaEnv.vector_intersection(
      self.location,
      v,
      self.map[self.next_checkpoint][2:4],
      w
      ) is None:
            centerline_distance += TrackmaniaEnv.norm(self.map_centerline[self.next_checkpoint] - self.prev_projection)
            self.prev_projection = self.map_centerline[self.next_checkpoint]
            self.next_checkpoint += 1
            if self.next_checkpoint >= len(self.map_centerline):
              self.next_checkpoint = 1
              self.rspwn = True
              self.done = True
            w = self.map[self.next_checkpoint][0:2] - self.map[self.next_checkpoint][2:4]

    # projection on the centerline
    '''k = self.map_centerline[self.next_checkpoint]-self.map_centerline[self.next_checkpoint-1]
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
    '''
    projection = TrackmaniaEnv.normal_projection(
      self.map_centerline[self.next_checkpoint],
      self.map_centerline[self.next_checkpoint - 1],
      self.location
    )

    centerline_distance += TrackmaniaEnv.norm(self.prev_projection - projection)
    self.prev_projection = projection

    # check wall contact
    penalty = 0.0
    if (self.view_buffer[-self.wall_number:] < TrackVisualizer.CONTACT_THRESHOLD).any():
      penalty = (self.speed_buffer[-1]**2) * 225.0

    # check stopped
    if self.step_counter > 10 and self.speed_buffer[-1] < 0.001:
      centerline_distance -= 500.0
      self.rspwn = True
      self.done = True
    
    self.done |= bool(self.data_getter.game_data[GameDataGetter.I_FINISH])
    self.rspwn |= self.done

    return centerline_distance - penalty


  def reset(self):
    self.step_counter = 0
    self.done = False
    if self.rspwn:
      self.rspwn = False
      self.respawn()

    return self.obs_buffer


  def step(self, action):
    self.apply_action(action)
    self.refresh_observation()
    reward = self.calc_reward()

    self.step_counter += 1

    # if self.rspwn:
    #   self.rspwn = False
    #   self.respawn()
    
    return self.obs_buffer, reward, self.done, {}

if __name__ == '__main__':
  env = TrackmaniaEnv('.\\Maps\\TurboTrack.Map.txt', human_driver=False)
  big = []
  for i in range(2):
    print(env.flip)
    env.reset()
    start = time.time()
    while not env.step([0,0])[2]:
      big.append(time.time()-start)
      print(env.view_buffer[-1])
      start = time.time()
      continue
  print(sum(big)/len(big))