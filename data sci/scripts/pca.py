import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io



def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data, None
    except Exception as e:
        return None, str(e)
    
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

def execute_full_pca(file_path, n_components=2):
    """Execute all PCA steps in sequence."""
    data = load_data(file_path)
    principal_df, pca = perform_pca(data, n_components=n_components)
    pca_plot = plot_pca(principal_df)
    return pca_plot, principal_df

