import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import io


def read_data(file_stream):
    """ Reads data from a file stream. """
    return pd.read_csv(file_stream)

def preprocess_data(data):
    """ Encode categorical variables and scale data. """
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, label_encoders

def perform_lda(data, target_column, n_components=2):
    """ Perform LDA and return the components. """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    lda = LDA(n_components=n_components)
    lda_components = lda.fit_transform(X, y)
    columns = [f'LDA Component {i+1}' for i in range(n_components)]
    lda_df = pd.DataFrame(data=lda_components, columns=columns)
    return lda_df, lda

def plot_lda(lda_df):
    """ Plot the LDA components. """
    plt.figure(figsize=(8, 6))
    plt.scatter(lda_df.iloc[:, 0], lda_df.iloc[:, 1] if lda_df.shape[1] > 1 else [0]*len(lda_df), alpha=0.5)
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2' if lda_df.shape[1] > 1 else '')
    plt.title('LDA Visualization')
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def execute_full_lda(file_stream, target_column, n_components=2):
    """ Execute all LDA steps in sequence. """
    data = read_data(file_stream)
    data, label_encoders = preprocess_data(data)
    lda_df, lda = perform_lda(data, target_column, n_components)
    lda_plot = plot_lda(lda_df)
    return lda_plot