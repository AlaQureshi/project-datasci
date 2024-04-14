import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data, None
    except Exception as e:
        return None, str(e)

def preprocess_data(data):
    if data is None:
        return None, "Data loading failed"
    try:
        data.ffill(inplace=True)  # Fill missing values
        return data, None
    except Exception as e:
        return None, str(e)

def generate_plots(data):
    if data is None:
        return None, "No data available for plotting"
    try:
        
        plots_info = []
        num_cols = data.select_dtypes(include=['number']).columns
        cat_cols = data.select_dtypes(include=['object', 'category']).columns

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
            pass  # Existing numeric plotting logic here

        for col in cat_cols:
            plt.figure(figsize=(10, 4))
            sns.countplot(x=col, data=data)  # Correct for categorical data
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


def execute_full_visualization(filepath):
    data, error = load_data(filepath)
    if error:
        return None, error

    data, error = preprocess_data(data)
    if error:
        return None, error

    plot_name = "visualization_plot"
    plot_path = save_plot(data, plot_name, output_directory="saved")
    return plot_path, "Plot generated and saved successfully"

def save_plot(data, plot_name, output_directory="saved"):
    os.makedirs(output_directory, exist_ok=True)
    plt.figure()
    plt.plot(data)  # Assuming data is properly formatted for plotting
    plt.title("Example Plot")
    plot_path = os.path.join(output_directory, f"{plot_name}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path
