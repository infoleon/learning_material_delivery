

import numpy as np
from datetime import datetime



def filter_vegetation_ind(threshold, matrix, dates, mode='from_first'):

    """
    Filters time steps (columns) in the NDVI matrix where any value is below the threshold.

    Parameters:
    - threshold (float): The minimum acceptable value.
    - matrix (np.ndarray): A 2D numpy array of shape (locations, time).
    - dates (list of str): A list of date strings, length must match the time dimension.

    Returns:
    - filtered_matrix (np.ndarray): NDVI matrix with only valid time steps.
    - filtered_dates (list of datetimes): Filtered list of dates matching the matrix.
    
    """
    # Ensure inputs are valid
    if matrix.shape[1] != len(dates):
        raise ValueError("The number of columns in matrix must match the length of the date list.")
    
    # Find time steps (columns) where all locations have NDVI >= threshold
    valid_mask = np.all(matrix >= threshold, axis=0)
    
    # Get the index to start filtering from
    indices = np.where(valid_mask)[0]
    if len(indices) == 0:
        return np.empty((matrix.shape[0], 0)), []  # nothing valid
    
    start_idx = indices[0] if mode == 'from_first' else indices[-1]
    
    # Slice from start_idx to end
    filtered_matrix = matrix[:, start_idx:]
    
    filtered_dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates[start_idx:]]

    return filtered_matrix, filtered_dates
















