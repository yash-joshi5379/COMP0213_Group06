from objects.Cube import baseCube


class CubeSmall(baseCube):
    """
    A small cube object for grasp planning simulation.

    This class represents a small cube with predefined dimensions and properties.
    It inherits from baseCube and initializes the object with parameters specific to
    the small cube variant.
    """

    def __init__(self):
        """
        Initialize a small cube object.

        Sets up the small cube with a URDF file reference, green color inherited
        from baseCube, initial height of 0.3 units, and an approach distance of
        0.35 units for grasping.
        """
        super().__init__(
            urdf_file="./urdf_files/objects/cubes/cube_small.urdf",
            colour=self.cube_colour,
            init_height=0.3,
            approach_dist=0.35)
