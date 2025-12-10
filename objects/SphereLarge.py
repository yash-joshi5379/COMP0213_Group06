from objects.Sphere import baseSphere


class SphereLarge(baseSphere):
    """
    A large sphere object for grasp planning simulation.

    This class represents a large sphere with predefined dimensions and properties.
    It inherits from baseSphere and initializes the object with parameters specific to
    the large sphere variant.
    """

    def __init__(self):
        """
        Initialize a large sphere object.

        Sets up the large sphere with a URDF file reference, blue color inherited
        from baseSphere, initial height of 0.11 units, and an approach distance of
        0.14128 units for grasping.
        """
        super().__init__(
            urdf_file="./urdf_files/objects/spheres/sphere_large.urdf",
            colour=self.sphere_colour,
            init_height=0.11,
            approach_dist=0.14128)
