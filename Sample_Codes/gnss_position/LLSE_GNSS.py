import numpy as np

# satellite position:meters
satellite_positions = np.array([
    [-13186870.6, 11385729.2, 19672626.3],
    [-7118031.6, 23256076.0, -9700477.9],
    [-2303925.9, 17164155.9, 20120354.5],
    [-15426414.5, 2696509.3, 22137570.3]
])

# satellite clock bias:meters
satellite_clock_bias = np.array([198812.8, 52245.17, 21575.56, 37173.51])

# ionospheric delay:meters
ionospheric_delay = np.array([3.8639, 4.6762, 3.614, 5.9277])

# tropospheric delay:meters
tropospheric_delay = np.array([3.24, 4.32, 3.07, 5.60])

pseudoranges_meas = np.array([21196662.1, 22222028.54, 21431397.16, 23928467.12])

# Set initial value
receiver_position = np.array([0.0, 0.0, 0.0])

receiver_clock_bias = 0



"""Calculate pseudo distance
Parameters:
satellite_positions - A three-dimensional array of satellite positions
receiver_position - Array of receiver positions
clock_bias - The value of the clock deviation 
ionospheric_delay - The value of the ionospheric delay (manually entered)
tropospheric_delay - The value of tropospheric delay (manual input)"""
def calculate_pseudoranges(satellite_positions, receiver_position, receiver_clock_bias, satellite_clock_bias, ionospheric_delay, tropospheric_delay):
    estimated_distances = np.linalg.norm(satellite_positions - receiver_position, axis=1)
    # Added receiver clock deviation, satellite clock deviation, ionospheric delay, and tropospheric delay
    pseudoranges = estimated_distances + receiver_clock_bias - satellite_clock_bias + ionospheric_delay + tropospheric_delay

    return pseudoranges

""" Solution of receiver position by least squares """
def least_squares_solution(satellite_positions, receiver_position, receiver_clock_bias, pseudoranges_meas):
    for _ in range(20):
        pseudoranges = calculate_pseudoranges(satellite_positions, receiver_position, receiver_clock_bias, satellite_clock_bias, ionospheric_delay, tropospheric_delay)
        pseudoranges_diff = pseudoranges_meas - pseudoranges

        # Initialize the matrix G
        G = np.zeros((len(satellite_positions), 4))

        # Calculate the matrix G
        for i in range(len(satellite_positions)):
            p_i = satellite_positions[i] - receiver_position
            r_i = np.linalg.norm(p_i)
            G[i, :3] = -p_i / r_i
            G[i, 3] = 1.0

        # Solve using least square method
        #delta_p = np.linalg.inv(G.T @ G) @ G.T @ pseudoranges_diff
        delta_p = np.linalg.lstsq(G, pseudoranges_diff, rcond=None)[0]
        receiver_position += delta_p[:3]
        receiver_clock_bias = delta_p[3]


        print(f"Iteration time =  {_}")
        print(f"pseudoranges =  {pseudoranges}")
        print(f"G =  {G}")
        print(f"delta_p =  {delta_p[:3]}")
        print(f"Estimated Receiver Position: {receiver_position}")
        print(f"Estimated Receiver Clock Bias: {receiver_clock_bias}")

        if np.linalg.norm(delta_p[:3]) < 1e-4:
            break
    return receiver_position, receiver_clock_bias


# Use the least square method to solve the receiver position
estimated_position, estimated_clock_bias = least_squares_solution(satellite_positions, receiver_position, receiver_clock_bias, pseudoranges_meas)

print(f"Estimated Receiver Position: {estimated_position}")
print(f"Estimated Clock Bias: {estimated_clock_bias}")
