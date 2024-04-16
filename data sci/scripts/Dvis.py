import pandas as pd
import matplotlib.pyplot as plt

# Function to load data
def load_data(filepath):
    return pd.read_csv(filepath)

# Function to plot numerical data as histograms
def plot_numerical_histograms(data, columns_to_exclude):
    for column in data.select_dtypes(include='number').columns:
        if column not in columns_to_exclude:
            plt.figure()
            data[column].dropna().hist()
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

# Function to plot numerical data as scatter plot
def plot_numerical_scatter(data, x_column, y_column):
    if pd.api.types.is_numeric_dtype(data[x_column]) and pd.api.types.is_numeric_dtype(data[y_column]):
        plt.figure()
        plt.scatter(data[x_column], data[y_column], alpha=0.5)
        plt.title(f'Scatter Plot of {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

# Function to plot categorical data
def plot_categorical_data(data):
    for column in data.select_dtypes(include='object').columns:
        plt.figure()
        data[column].value_counts().plot(kind='bar')
        plt.title(f'Bar Chart of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

# Main function to process and plot data
def process_and_plot_data(filepath, x_column=None, y_column=None):
    data = load_data(filepath)
    
    if x_column and y_column:
        plot_numerical_scatter(data, x_column, y_column)
        plot_numerical_histograms(data, {x_column, y_column})
    else:
        plot_numerical_histograms(data, set())

    plot_categorical_data(data)



# Example usage
#process_and_plot_data(r'C:\Users\alaqu\OneDrive\Desktop\project-datasci\test values\discrete_data.csv', x_column='Age', y_column='Income')



