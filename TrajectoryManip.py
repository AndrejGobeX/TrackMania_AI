import numpy as np

from imitation.data.types import TrajectoryWithRew


class TrajectoryManip():
    def __init__(self):
        pass


    def load_trajectory(filename:str)->TrajectoryWithRew:
        file = np.load(filename, allow_pickle=True)
        trajectory = TrajectoryWithRew(
            file['obs'],
            file['act'],
            file['inf'],
            True,
            file['rews']
        )
        return trajectory
