from objects.baseObject import baseObject
import numpy as np


class baseCube(baseObject):
    """
    A cube object for grasp planning simulation.

    This class represents a cube-shaped object with a predefined green color.
    It inherits from baseObject and implements the plot method to visualize the cube
    as a wireframe with vertices and faces.
    """
    cube_colour = [0, 1, 0, 1]

    def plot(self):
        """
        Generate the wireframe representation of the cube for visualization.

        Creates a cube centered at (cx, cy, cz) with side length 2*s. Computes
        all 8 vertices and 6 faces of the cube.

        Returns:
            list: A list of 6 faces, where each face is a list of 4 vertices
                that define the corners of that face.
        """
        cx, cy, cz = self.cx, self.cy, self.cz
        s = self.s

        vertices = np.array([
            [cx - s, cy - s, cz - s],
            [cx + s, cy - s, cz - s],
            [cx + s, cy + s, cz - s],
            [cx - s, cy + s, cz - s],
            [cx - s, cy - s, cz + s],
            [cx + s, cy - s, cz + s],
            [cx + s, cy + s, cz + s],
            [cx - s, cy + s, cz + s],
        ])

        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],
            [vertices[j] for j in [4, 5, 6, 7]],
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [2, 3, 7, 6]],
            [vertices[j] for j in [1, 2, 6, 5]],
            [vertices[j] for j in [4, 7, 3, 0]],
        ]

        return faces
