import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Use a non-interactive backend to avoid GUI usage from worker threads.
matplotlib.use("Agg")


class Labeler:
    """
    A utility class for collecting, labeling, and visualizing grasp trial data.

    This class stores feature vectors and labels, saves them to CSV files,
    and generates 3D visualizations of the grasp attempts with success/failure indicators.
    """

    def __init__(self):
        """
        Initialize the Labeler with empty DataFrames for features and labels.

        Sets up data structures to store trial data with columns for position
        (x, y, z) and orientation (roll, pitch, yaw), as well as binary labels
        indicating grasp success or failure.
        """
        # Initialize empty DataFrames with proper columns
        self.feature_columns = ["x", "y", "z", "roll", "pitch", "yaw"]
        self.data = pd.DataFrame(columns=self.feature_columns)
        self.labels = pd.DataFrame(columns=["label"])

    def add_trial_data(self, trial_data):
        """
        Add feature data from a single grasp trial.

        Appends a new row of trial data containing position and orientation information
        to the internal data DataFrame.

        Args:
            trial_data (dict or array-like): A row of data containing [x, y, z, roll, pitch, yaw] values.
        """
        df_row = pd.DataFrame([trial_data], columns=self.data.columns)
        if self.data.empty:
            self.data = df_row
        else:
            self.data = pd.concat([self.data, df_row], ignore_index=True)

    def add_trial_label(self, label):
        """
        Add a label for a grasp trial indicating success or failure.

        Appends a new label (1 for success, 0 for failure) to the
        internal labels DataFrame.

        Args:
            label (int): Binary label indicating grasp outcome (0 or 1).
        """
        df_label = pd.DataFrame([{"label": label}])
        if self.labels.empty:
            self.labels = df_label
        else:
            self.labels = pd.concat([self.labels, df_label], ignore_index=True)

    def save_dataset(self, save_data_name):
        """
        Combine and save the collected trial data and label it to a CSV file.

        Concatenates the feature data and labels into a single DataFrame and saves
        it to a CSV file in the 'datasets' directory.

        Args:
            save_data_name (str): The name (without extension) for the output CSV file.
        """
        self.combined_data = pd.concat([self.data, self.labels], axis=1)
        # print(self.combined_data.head())

        os.makedirs("datasets", exist_ok=True)
        save_file_name = os.path.join("datasets", save_data_name + ".csv")
        self.combined_data.to_csv(save_file_name, index=False)
        print("Data saved")

    def plot_grasp_results(self, init_dir_vecs,
                           save_data_name, object_code, object):
        """
        Generate and save a 3D visualization of grasp attempts and results.

        Creates a 3D plot showing the direction vectors for each grasp attempt (green = success, red = failure), and visualizes the target object in the scene.
        Saves the figure as a PNG file in the 'figures' directory.

        Args:
            init_dir_vecs (np.ndarray): Array of initial direction vectors with shape
                (n_trials, 3) containing [dx, dy, dz] for each trial.
            save_data_name (str): The name (without extension) for the output PNG file.
            object_code (str): Code indicating object type ('B' for cube/block, 'S' for sphere).
            object: An object instance with a plot() method that returns either:
                - faces (for cubes): list of face vertices
                - (X, Y, Z) (for spheres): surface coordinates for plotting
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        colours = ['g' if self.combined_data.loc[i, 'label'] ==
                   1 else 'r' for i in range(len(self.combined_data))]
        # print(colours)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 3)

        ax.quiver(self.combined_data['x'],
                  self.combined_data['y'],
                  self.combined_data['z'],
                  init_dir_vecs[:,
                  0],
                  init_dir_vecs[:,
                  1],
                  init_dir_vecs[:,
                  2],
                  color=colours,
                  length=0.4)

        if object_code == "B":
            faces = object.plot()
            ax.add_collection3d(
                Poly3DCollection(
                    faces,
                    linewidths=1,
                    edgecolors="k"))
        else:
            X, Y, Z = object.plot()
            ax.plot_surface(X, Y, Z)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        legend_elements = [
            Line2D([0], [0], color='g', linewidth=3, label='Successful Grasp'),
            Line2D(
                [0],
                [0],
                color='r',
                linewidth=3,
                label='Unsuccessful Grasp')
        ]
        ax.legend(handles=legend_elements)

        os.makedirs("figures", exist_ok=True)
        save_image_name = os.path.join("figures", save_data_name + ".png")
        plt.savefig(save_image_name)
