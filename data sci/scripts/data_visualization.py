import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io

def load_data(file_stream):
    try:
        return pd.read_csv(file_stream), None
    except Exception as e:
        return None, str(e)

def preprocess_data(data):
    if data is None:
        return None, "Data loading failed"
    try:
        data.ffill(inplace=True)  # Corrected forward fill usage
        return data, None
    except Exception as e:
        return None, str(e)

def generate_plots(data):
    if data is None:
        return None, "No data available for plotting"
    try:
        plots_info = []
        num_cols = data.select_dtypes(include=['number']).columns
        cat_cols = data.select_dtypes(include=['object']).columns

        for col in num_cols:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            sns.histplot(data[col], kde=True)
            plt.title(f'Histogram of {col}')
            plt.subplot(1, 3, 2)
            sns.boxplot(x=data[col])
            plt.title(f'Boxplot of {col}')
            plt.subplot(1, 3, 3)
            sns.violinplot(x=data[col])
            plt.title(f'Violin plot of {col}')
            plt.tight_layout()
            
            plot_buf = io.BytesIO()
            plt.savefig(plot_buf, format='png')
            plot_buf.seek(0)
            plots_info.append((f'combined_plots_{col}', plot_buf.getvalue()))
            plt.close()

        for col in cat_cols:
            plt.figure(figsize=(10, 4))
            sns.countplot(x=col, data=data)
            plt.title(f'Count plot of {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_buf = io.BytesIO()
            plt.savefig(plot_buf, format='png')
            plot_buf.seek(0)
            plots_info.append((col, plot_buf.getvalue()))
            plt.close()

        if not plots_info:
            return None, "No plots were generated"
        return plots_info, "Plots generated successfully"
    except Exception as e:
        return None, str(e)

def execute_full_visualization(file_stream):
    data, error = load_data(file_stream)
    if error:
        return None, error

    processed_data, error = preprocess_data(data)
    if error:
        return None, error

    results, error = generate_plots(processed_data)
    if error:
        return None, error

    return results, "Plots generated successfully"