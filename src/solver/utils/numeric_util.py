def get_boundary_tolerance(value, lower_limit, upper_limit):
    """
    Calculate the boundary violation if the value is outside the specified limits.

    Args:
        value (float): The value to check.
        lower_limit (float): The minimum allowable value.
        upper_limit (float): The maximum allowable value.

    Returns:
        float: The calculated boundary violation if the value is out of bounds, otherwise 0.0.
    """
    tolerance = 0.0  # Initialize the boundary tolerance to 0

    if value < lower_limit:
        tolerance = lower_limit - value
    elif value > upper_limit:
        tolerance = value - upper_limit

    return tolerance


def get_deviation(value, setpoint, threshold):
    """
    Calculate the deviation of a value from its setpoint.

    Args:
        value (float): The current value to check.
        setpoint (float): The target setpoint value.
        threshold (float, optional): An allowable tolerance range. Defaults to 0.

    Returns:
        float: The deviation from the setpoint. Returns 0 if within the tolerance threshold.
    """
    tolerance = abs(value - setpoint)

    if tolerance <= threshold:
        return 0.0
    else:
        return tolerance