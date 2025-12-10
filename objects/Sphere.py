from objects.baseObject import baseObject
import numpy as np


class baseSphere(baseObject):
    """
    A sphere object for grasp planning simulation.

    This class represents a spherical object with a predefined blue color.
    It inherits from baseObject and implements the plot method to visualize the sphere
    using a parametric surface representation.
    """
    sphere_colour = [0, 0, 1, 1]

    def plot(self):
        """
        Generate the surface representation of the sphere for visualization.

        Creates a sphere centered at (cx, cy, cz) with radius s using parametric
        equations. Generates a mesh grid representing the sphere's surface using
        azimuthal and polar angles.

        Returns:
            tuple: A tuple of three numpy arrays (X, Y, Z) representing the 3D
                coordinates of the sphere surface, suitable for 3D plotting.
        """
        cx, cy, cz = self.cx, self.cy, self.cz
        s = self.s

        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        U, V = np.meshgrid(u, v)
        X = cx + s * np.cos(U) * np.sin(V)
        Y = cy + s * np.sin(U) * np.sin(V)
        Z = cz + s * np.cos(V)

        return X, Y, Z
