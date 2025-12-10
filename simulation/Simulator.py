import pybullet as p
import pybullet_data
import numpy as np
from objects.CubeLarge import CubeLarge
from objects.SphereLarge import SphereLarge
from objects.CubeSmall import CubeSmall
from objects.SphereSmall import SphereSmall
from grippers.ThreeFingerHand import ThreeFingerHand
from grippers.TwoFingerHand import TwoFingerHand
from simulation.Labeler import Labeler
import time


class Simulator:
    """
    A simulation environment for grasp experiments.

    This class manages PyBullet-based simulations for testing different gripper
    configurations (2-finger or 3-finger) on various objects (cubes or spheres).
    It handles the complete grasp pipeline including gripper orientation, approach,
    grasping, lifting, and stability evaluation. Results are collected and saved
    for machine learning analysis.
    """
    gripper_map = {3: ThreeFingerHand,
                   2: TwoFingerHand}

    object_map = {(3, "B"): CubeLarge,
                  (3, "S"): SphereLarge,
                  (2, "B"): CubeSmall,
                  (2, "S"): SphereSmall, }

    def __init__(self):
        """
        Initialize the simulation environment.

        Sets up the PyBullet physics simulation with GUI, configures the camera,
        loads the ground plane, sets gravity, and initializes a Labeler for
        collecting trial data. Hides various debug preview windows for cleaner visualization.
        """
        cid = p.connect(p.SHARED_MEMORY)
        if cid < 0:
            p.connect(p.GUI)
            # hide right "Params" tab
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(
                p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                0)         # hide RGB preview
            p.configureDebugVisualizer(
                p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                0)       # hide depth preview
            p.configureDebugVisualizer(
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0)  # hide segmentation preview

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()

        # Set ground plane
        p.loadURDF("plane.urdf")

        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.1, -0.1, 0.3]
        )

        self.lab = Labeler()
        self.init_dir_vecs = np.empty((0, 3), dtype=np.float64)

    @staticmethod
    def draw_axes():
        """
        Draw X, Y, Z coordinate axes in the simulation environment.

        Renders red, green, and blue lines representing the X, Y, and Z axes
        respectively. Also adds text labels for each axis.
        """
        # Plot x,y,z axes and labels
        p.addUserDebugLine([0, 0, 0], [0.5, 0, 0], [1, 0, 0], 10, 0)
        p.addUserDebugLine([0, 0, 0], [0, 0.5, 0], [0, 1, 0], 10, 0)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.5], [0, 0, 1], 10, 0)
        p.addUserDebugText('x', [0.5, 0, 0], [0, 0, 0])
        p.addUserDebugText('y', [0, 0.5, 0], [0, 0, 0])
        p.addUserDebugText('z', [0, 0, 0.5], [0, 0, 0])

    def run_simulation(self, gripper, object_type):
        """
        Execute a complete grasp simulation trial.

        Runs through the full grasp pipeline: initializes the gripper and object,
        orients the gripper towards the object, opens the gripper, approaches the object,
        closes the gripper, and lifts the object.
        Initial positions and orientations are recorded for later analysis.

        Args:
            gripper (int): Number of fingers for the gripper (2 or 3).
            object_type (str): Type of object to grasp ('B' for cube/block, 'S' for sphere).
        """
        self.draw_axes()
        self.num_fingers, self.object_code = gripper, object_type
        self.gripper = Simulator.gripper_map[gripper]()
        self.object_type = Simulator.object_map[gripper, object_type]()

        # Orient gripper towads object
        for _ in range(480):
            self.gripper.orient_towards_object(self.object_type)
            p.stepSimulation()
            time.sleep(1. / 240.)

        # Put starting position and orientation into a 2d list
        init_pos, init_ori = p.getBasePositionAndOrientation(
            self.gripper.gripper_id)
        init_conditions = [*init_pos, *p.getEulerFromQuaternion(init_ori)]
        self.lab.add_trial_data(init_conditions)

        # Find initial orientation as a direction vector (used for plotting
        # later)
        init_dir_vec, _, _ = self.gripper.get_direction_vector(
            self.object_type)
        init_dir_vec = init_dir_vec / np.linalg.norm(init_dir_vec)
        init_dir_vec = np.asarray(init_dir_vec, dtype=float).reshape(1, 3)
        self.init_dir_vecs = np.concatenate(
            [self.init_dir_vecs, init_dir_vec], axis=0)

        # Get finger positions ready to grab object
        self.gripper.preshape()

        # Open gripper
        self.gripper.open_gripper()

        # Go towards object
        for _ in range(1200):
            continue_moving = self.gripper.move_to_object(
                self.object_type, self.object_type.approach_dist)
            if continue_moving == 0:
                break
            p.stepSimulation()
            time.sleep(1. / 240.)

        # Close gripper around object to grab objects
        self.gripper.close_gripper()
        p.stepSimulation()
        time.sleep(1)

        # Move gripper and object up together
        for _ in range(480):
            self.gripper.move_up()
            p.stepSimulation()
            time.sleep(1. / 240.)

    def classify_grasp(self):
        """
        Evaluate and classify the grasp as successful or unsuccessful.

        Checks if the grasped object is stable after being lifted by calling the
        object's check_stability() method. Records the binary result (1 for success,
        0 for failure) in the labeler.

        Returns:
            bool: True if the grasp is stable and successful, False otherwise.
        """
        # Check if grasp is stable
        result = self.object_type.check_stability()
        self.lab.add_trial_label(int(result))
        return result

    @staticmethod
    def reset_simulation():
        """
        Reset the simulation environment to its initial state.

        Clears all objects from the simulation, reloads the ground plane,
        and resets gravity. Used between trials to ensure a clean starting state.
        """
        p.resetSimulation()
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

    def end_simulation(self):
        """
        Finalize the simulation and save all collected data.

        Constructs a filename based on the gripper type, object type, and number
        of trials, then saves the dataset to CSV, disconnects from PyBullet,
        and uses the Labeler to visualize results.
        """
        save_data_name = f"NumFingers_{
            self.num_fingers}_Object_{
            self.object_code}_NumTrials_{
            len(
                self.lab.data)}"
        self.lab.save_dataset(save_data_name)
        print("Results saved")
        p.disconnect()
        self.lab.plot_grasp_results(
            self.init_dir_vecs,
            save_data_name,
            self.object_code,
            self.object_type)
