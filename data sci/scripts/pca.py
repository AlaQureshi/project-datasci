import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io



def read_data(file_stream):
    """ Reads data from a file stream and returns a DataFrame. """
    return pd.read_csv(file_stream)

def perform_pca(data, n_components=2):
    """ Performs PCA on the provided DataFrame. """
    # Standardizing the features
    data = StandardScaler().fit_transform(data)

    # PCA transformation
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)

    # Creating a DataFrame with principal components
    columns = [f'Principal Component {i+1}' for i in range(n_components)]
    principal_df = pd.DataFrame(data=principal_components, columns=columns)
    return principal_df, pca

def plot_pca(principal_df):
    """ Plots the first two principal components of the PCA result. """
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_df['Principal Component 1'], principal_df['Principal Component 2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2 Component PCA')
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def execute_full_pca(file_stream, n_components=2):
    """Execute all PCA steps in sequence."""
    data = read_data(file_stream)
    principal_df, pca = perform_pca(data, n_components=n_components)
    pca_plot = plot_pca(principal_df)
    return pca_plot, principal_df

