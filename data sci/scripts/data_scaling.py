import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def load_data(file):
    """Load DataFrame from file, returning the data and an error message if applicable."""
    try:
        data = pd.read_csv(file)
        return data, None  # Return None for the error if loading is successful
    except Exception as e:
        return None, str(e)  # Return None for the data and the error message

def detect_numeric_columns(data):
    """Detect and return numeric columns from the DataFrame."""
    if data is None:
        return None  # If data is None, return None for numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    return numeric_cols

def scale_data(data, columns, method='standard'):
    """Scale data based on the selected method."""
    if data is None or columns is None:
        return None  # Return None if data or columns are None
    scaler = get_scaler(method)
    data[columns] = scaler.fit_transform(data[columns])
    return data

def get_scaler(method):
    """Return the appropriate scaler based on the method."""
    if method == 'minmax':
        return MinMaxScaler()
    elif method == 'standard':
        return StandardScaler()
    elif method == 'robust':
        return RobustScaler()
    return StandardScaler()  # Default to StandardScaler if method is not recognized

def execute_full_scaling(filepath, method='standard'):
    """Execute all steps for data scaling on the given file."""
    data, error = load_data(filepath)
    if error:
        return None, error  # Return immediately if there is an error loading data
    numeric_cols = detect_numeric_columns(data)
    scaled_data = scale_data(data, numeric_cols, method)
    if scaled_data is None:
        return None, "Error in scaling data"  # Handle errors in scaling
    return scaled_data, None  # Return scaled data and None for error if successful

