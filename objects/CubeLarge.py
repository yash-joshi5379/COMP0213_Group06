from objects.Cube import baseCube


class CubeLarge(baseCube):
    """
    A large cube object for grasp planning simulation.

    This class represents a large cube with predefined dimensions and properties.
    It inherits from baseCube and initializes the object with parameters specific to
    the large cube variant.
    """

    def __init__(self):
        """
        Initialize a large cube object.

        Sets up the large cube with a URDF file reference, green color inherited
        from baseCube, initial height of 0.3 units, and an approach distance of
        0.17 units for grasping.
        """
        super().__init__(
            urdf_file="./urdf_files/objects/cubes/cube_large.urdf",
            colour=self.cube_colour,
            init_height=0.3,
            approach_dist=0.17)
