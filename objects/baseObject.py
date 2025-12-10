import pybullet as p
import numpy as np
import time
from abc import ABC, abstractmethod


class baseObject(ABC):
    """
    Abstract base class for objects in the grasp planning simulation.

    This class provides common functionality for simulated objects including loading
    URDF models, managing physical properties, checking stability, and visualizing
    objects in the simulation environment.
    """

    def __init__(self, urdf_file, colour, init_height, approach_dist):
        """
        Initialize the baseObject.

        Args:
            urdf_file (str): Path to the URDF file defining the object geometry and dynamics.
            colour (tuple): RGBA color tuple (r, g, b, a) for the object's visual appearance.
            init_height (float): Initial height (z-coordinate) of the object above the ground.
            approach_dist (float): The desired approach distance for the gripper when grasping this object.
        """
        self.urdf_file, self.colour, self.height, self.approach_dist = urdf_file, colour, init_height, approach_dist
        self.position = [0, 0, self.height]
        self.orientation = [0, 0, 0, 1]
        self.body_id = p.loadURDF(self.urdf_file,
                                  basePosition=self.position,
                                  baseOrientation=self.orientation)
        p.changeVisualShape(self.body_id, -1, rgbaColor=self.colour)

        # Define centre coordinates and size for plotting later
        self.cx, self.cy, self.cz = [0, 0, 0.3]
        self.s = 0.3

    def check_stability(self):
        """
        Check if the object is stable after being grasped and lifted.

        Waits 3 seconds for the object to settle, then measures the distance it has
        moved and its height. The object is considered stable if it has moved less
        than 0.01 units (stationary) and is still above 0.15 units in height (off the ground).

        Returns:
            bool: True if the object is stable and above the minimum height, False otherwise.
        """
        current_obj_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        current_obj_pos = np.array(current_obj_pos)
        time.sleep(3)
        new_obj_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        new_obj_pos = np.array(new_obj_pos)
        abs_diff = np.linalg.norm(current_obj_pos - new_obj_pos)
        # print(abs_diff)
        return abs_diff < 0.01 and new_obj_pos[2] > 0.15

    @abstractmethod
    def plot(self):
        """
        Plot or visualize the object. Must be implemented by subclasses.
        """
        pass
