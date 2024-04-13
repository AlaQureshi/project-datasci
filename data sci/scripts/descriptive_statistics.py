import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


def read_data(file_stream):
    """Read data from a file stream."""
    return pd.read_csv(file_stream)

def generate_statistics(data):
    """Generate descriptive statistics and visualizations."""
    results = {}
    buf = io.BytesIO()
    
    # Descriptive statistics for numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        desc_stats = numeric_data.describe()
        skewness = numeric_data.skew()
        kurtosis = numeric_data.kurt()
        desc_stats.loc['skew'] = skewness
        desc_stats.loc['kurtosis'] = kurtosis
        results['numeric_stats'] = desc_stats.to_json()

        # Correlation matrix
        correlation_matrix = numeric_data.corr()
        results['correlation_matrix'] = correlation_matrix.to_json()

        # Pairplot for visual inspection of data
        sns.pairplot(numeric_data)
        plt.savefig(buf, format='png')
        buf.seek(0)
        results['pairplot'] = buf
    
    # Descriptive statistics for categorical columns
    categorical_data = data.select_dtypes(include=['object'])
    categorical_stats = {}
    if not categorical_data.empty:
        for column in categorical_data:
            value_counts = categorical_data[column].value_counts().to_string()
            column_desc = categorical_data[column].describe().to_string()
            categorical_stats[column] = {'value_counts': value_counts, 'describe': column_desc}
        results['categorical_stats'] = categorical_stats
    
    return results

def execute_full_statistics(file_stream):
    """Execute all descriptive statistics generation steps in sequence."""
    data = read_data(file_stream)
    results = generate_statistics(data)
    return results