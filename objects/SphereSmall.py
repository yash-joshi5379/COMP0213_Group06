from objects.Sphere import baseSphere


class SphereSmall(baseSphere):
    """
    A small sphere object for grasp planning simulation.

    This class represents a small sphere with predefined dimensions and properties.
    It inherits from baseSphere and initializes the object with parameters specific to
    the small sphere variant.
    """

    def __init__(self):
        """
        Initialize a small sphere object.

        Sets up the small sphere with a URDF file reference, blue color inherited
        from baseSphere, initial height of 0.07 units, and an approach distance of
        0.2938 units for grasping.
        """
        super().__init__(
            urdf_file="./urdf_files/objects/spheres/sphere_small.urdf",
            colour=self.sphere_colour,
            init_height=0.07,
            approach_dist=0.2938)
