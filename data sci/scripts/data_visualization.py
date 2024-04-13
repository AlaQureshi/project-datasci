import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def load_data(file_stream):
    try:
        data = pd.read_csv(file_stream)
        return data
    except Exception as e:
        return None, str(e)

def preprocess_data(data):
    if data is None:
        return None, "No data loaded"
    try:
        output = io.StringIO()
        data.info(buf=output)
        info = output.getvalue()
        output.close()

        if data.isnull().sum().sum() > 0:
            data.fillna(data.mean(), inplace=True)
            for col in data.select_dtypes(include='object').columns:
                data[col].fillna(data[col].mode()[0], inplace=True)
            info += '\nMissing values filled.'
        else:
            info += '\nNo missing values detected.'
        return data, info
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

        return plots_info, None
    except Exception as e:
        return None, str(e)

def execute_full_visualization(file_stream):
    data, error = load_data(file_stream)
    if error:
        return None, error

    processed_data, info = preprocess_data(data)
    if processed_data is None:
        return None, info

    result, error = generate_plots(processed_data)
    if error:
        return None, error

    return result, info
