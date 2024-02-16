from gymnasium import Env, spaces

import numpy as np
import math
import time
import cv2

from stable_baselines3.common.env_checker import check_env

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

# *** UTILITY FNS ***
def dot(x:np.array, y:np.array) -> float:
    return x[0]*y[0]+x[1]*y[1]+x[2]*y[2]


def point_2_line_projection(a:np.array, b:np.array, p:np.array) -> np.array:
    ap = p-a
    ab = b-a
    result = a + dot(ap,ab)/dot(ab,ab) * ab
    return result


def get_3_point_ratio(a:np.array, b:np.array, p:np.array) -> float:
    dx = b[0]-a[0]
    dxp = p[0]-a[0]
    if dx == 0:
        dx = b[1]-a[1]
        dxp = p[1]-a[1]
    return dxp/dx


def find_point_by_lenght(lengths:np.array, i:int, distance:float, start_offset:float = 0.0) -> (int, float):
    distance += start_offset
    
    while i < len(lengths) and distance > lengths[i]:
        distance -= lengths[i]
        i += 1
    
    return (i, distance)
    

# *** MAIN ***
class TMCamEnv(Env):

    PENALTY_WRONG_WAY:float = -100.0
    LOOKAHEAD:float = 100.0
    
    def __init__(
            self,
            map_centerline_path:str,
            image_w:int=80,
            image_h:int=45,
            action_duration:float=0.05,
            obs_duration:float=0.0021,
            start_delay:float=1.9,
            human_driver:bool=False) -> None:
        """Gymnasium compatible env for reinforcement learning in Trackmania.
        
        ### Parameters
        map_centerline_path : str
            Path to `.npy` track centerline file.
            Numpy array should be in (-1, 3) format.
        image_w : int (default 50)
            Self-explanatory.
        image_h : int (default 50)
            Self-explanatory.
        action_duration : float, (default 0.05)
            Desired amount of time between obtaining an observation and sending an action in seconds.
            If it takes less than action_duration, the thread will wait the rest of the time.
            If it takes more, the thread will print a timeout.
        obs_duration : float, (default 0.0021)
            Desired amount of time between sending an action and obtaining an observation in seconds.
            If it takes less than obs_duration, the thread will wait the rest of the time.
            If it takes more, the thread will print a timeout.
        start_delay : float, (default 1.9)
            The amount of time to wait after each restart (waiting for the countdown).
            If training, leave the default value.
            If validating, set to 0.0
        human_driver : bool, (default False)
            If True, env will not apply any action.
        """
        super().__init__()

        self.image_w = image_w
        self.image_h = image_h
        self.centerline = np.load(map_centerline_path)
        self.centerline_deltas = self.centerline[1:] - self.centerline[:-1]
        self.action_duration = action_duration
        self.obs_duration = obs_duration
        self.start_delay = start_delay
        self.human_driver = human_driver
        
        self.observation_space = TMCamEnv.make_space(image_w, image_h)
        # steer, throttle/brake
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        # flipping mechanism
        self.flip = True

        # utils
        self.data_getter = GameDataGetter()
        self.window = WindowInterface()


    def make_space(w:int, h:int) -> spaces.Dict:
        image_space = TMCamEnv.make_image_space(w, h)
        scalar_space = TMCamEnv.make_scalar_space()
        dict_space = spaces.Dict({
            "img":image_space,
            "scalar":scalar_space
        })
        return dict_space


    def make_image_space(w:int, h:int) -> spaces.Box:
        # w*h grayscale image
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w),
            dtype=np.uint8
        )
        return image_space


    def make_scalar_space() -> spaces.Box:
        # speed, cp1, cp1, cp1, cp2, cp2, cp2, prev_steer, prev_throttle_brake
        scalar_space = spaces.Box(
            low=np.array([0, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        return scalar_space
    

    def compress_image(self, sch) -> np.array:
        sch = cv2.cvtColor(sch, cv2.COLOR_RGBA2GRAY)
        sch = cv2.resize(sch, (self.image_w, self.image_h))
        sch = cv2.convertScaleAbs(sch, alpha=1.7, beta=0)
        return sch

    
    def update_game_data(self) -> None:
        self.location = np.array([
            self.data_getter.game_data[GameDataGetter.I_X],
            self.data_getter.game_data[GameDataGetter.I_Z],
            self.data_getter.game_data[GameDataGetter.I_Y]
        ])
        self.speed = self.data_getter.game_data[GameDataGetter.I_SPEED] * 0.0036
        self.image = self.compress_image(self.window.screenshot())

    
    def respawn(self) -> None:
        """Respawns the car to the start.
        """
        tm_reset()
        tm_update()
        tm_respawn()
        if self.start_delay > 0.0:
            time.sleep(self.start_delay)
        
        self.update_game_data()

        self.prev_action = np.array([0,0])
        self.centerline_i = 0
        self.current_segment_coverage = 0.0
        self.progress()

        self.flip = not self.flip


    def progress(self) -> float:
        centerline_progress = 0.0

        while True:
            if self.centerline_i == len(self.centerline):
                return centerline_progress
        
            p = point_2_line_projection(
                self.centerline[self.centerline_i],
                self.centerline[self.centerline_i+1],
                self.location
            )
            coverage = get_3_point_ratio(
                self.centerline[self.centerline_i],
                self.centerline[self.centerline_i+1],
                p
            )

            # wrong way, reset
            if coverage < self.current_segment_coverage:
                self.done = True
                return TMCamEnv.PENALTY_WRONG_WAY
            
            # next segment
            if coverage >= 1.0:
                centerline_progress += \
                    (1.0-self.current_segment_coverage)*self.centerline_deltas[self.centerline_i]
                self.centerline_i += 1
                self.current_segment_coverage = 0.0
                continue

            centerline_progress += \
                (coverage-self.current_segment_coverage)*self.centerline_deltas[self.centerline_i]
            self.current_segment_coverage = coverage
            
            return centerline_progress

    
    def apply_action(self, action:np.ndarray):
        """Applies the given action.

        ### Parameters
        action : np.ndarray
            Array of two floats [-1,1] indicating, respectively, steering and throttle/braking.
        """
        # overwrite
        self.prev_action = action

        if self.flip:
            action[0] = -action[0]
        
        if not self.human_driver:
            tm_reset()

            tm_steer(action[0])
            if action[1] > 0.0:
                tm_accelerate(action[1])
            elif self.data_getter.game_data[GameDataGetter.I_GEAR] > 0.1 and self.speed_buffer[-1] > 0.02:
                tm_brake(-action[1])
        
            tm_update()
    
    
    def make_observation(self) -> dict:
        obs = {
            "img":self.image,
            #TODO: popravi
            "scalar":np.array([self.speed, 2, 2, 2, 3, 3, 3, self.prev_action[0], self.prev_action[1]])
        }
        return obs
    
    
    def reset(self, seed=None, options=None) -> tuple:
        self.done = False
        self.respawn()
        self.start_time = time.time()
        self.action_time = time.time()
        return self.make_observation(), {}


    def step(self, action):
        # action
        self.action_time = time.time() - self.action_time
        if self.action_duration > self.action_time:
            time.sleep(self.action_duration - self.action_time)
        else:
            print("Action timeout: ", self.action_time - self.action_duration)
        
        self.apply_action(action)

        # observation
        obs_time = time.time()
    
        self.update_game_data()
        reward = self.progress()
        find_point_by_lenght(self.centerline_deltas, self.centerline_i, )

        obs_time = time.time() - obs_time
        if self.obs_duration > obs_time:
            time.sleep(self.obs_duration - obs_time)
        else:
            print("Observation timeout: ", obs_time - self.obs_duration)

        # time next action
        self.action_time = time.time()
        
        return self.make_observation(), reward, self.done, False, {}

