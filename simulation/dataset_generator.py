from simulation.Simulator import Simulator


def generate_dataset(gripper, object_type, num_trials):
    """
    Dataset generation pipeline: creates a labeled dataset of grasps through simulations.

    Runs multiple grasp simulation trials with the specified gripper and object,
    evaluates each grasp for stability, collects position and orientation data,
    and saves the complete dataset with success/failure labels to a CSV file.

    Args:
        gripper (int): Number of fingers for the gripper (2 for two-finger, 3 for three-finger).
        object_type (str): Type of object to grasp ('B' for box/cube, 'S' for sphere).
        num_trials (int): Number of grasp trials to simulate and record.
    """
    num_successes = 0
    sim = Simulator()

    for i in range(num_trials):
        sim.run_simulation(gripper, object_type)
        result = sim.classify_grasp()

        if result == True:
            num_successes += 1

        print(f"Iteration {i + 1} : {result}")
        print(
            f"Successes = {num_successes} | Failures = {
                i + 1 - num_successes}")

        if i == num_trials - 1:
            sim.end_simulation()
        else:
            sim.reset_simulation()
