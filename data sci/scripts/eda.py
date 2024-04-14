import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data, None
    except Exception as e:
        return None, str(e)


def summarize_data(df):
    """ Generate summary statistics. """
    if isinstance(df, tuple):
        data, error = df
        if error is not None:
            return {'error': error}
        df = data
    data_summary = {
        'describe': df.describe().to_json(),
        'dtypes': df.dtypes.to_string(),
        'missing_values': df.isnull().sum().to_string(),
        'first_rows': df.head().to_string()
    }
    return data_summary

def plot_histograms(df):
    """ Generate histograms for numerical columns. """
    if isinstance(df, tuple):
        data, error = df
        if error is not None:
            return {'error': error}
        df = data
    num_cols = df.select_dtypes(include=['number']).columns
    fig, axes = plt.subplots(len(num_cols), 1, figsize=(10, len(num_cols) * 4))
    for ax, col in zip(axes.flatten(), num_cols):
        sns.histplot(df[col], bins=15, ax=ax)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def plot_correlation_matrix(df):
    """ Generate correlation matrix plot. """
    if isinstance(df, tuple):
        data, error = df
        if error is not None:
            return {'error': error}
        df = data
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def plot_missing_data_heatmap(df):
    """ Plot heatmap for missing data. """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Data Heatmap")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def execute_full_eda(file_path):
    """Execute all EDA steps in sequence."""
    data = load_data(file_path)
    summary = summarize_data(data)
    histograms = plot_histograms(data)
    correlation_matrix = plot_correlation_matrix(data)
    missing_data_heatmap = plot_missing_data_heatmap(data)
    return {
        "summary": summary,
        "histograms": histograms,
        "correlation_matrix": correlation_matrix,
        "missing_data_heatmap": missing_data_heatmap
    }
