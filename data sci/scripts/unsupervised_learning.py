import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import io


def load_data(file_stream):
    """Load data from a file depending on the file type."""
    
    return pd.read_csv(file_stream)
    

def preprocess_data(data, file_type):
    """Preprocess data based on its type."""
    if file_type == 'csv':
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.select_dtypes(include=[float, int]))
        return scaled_data
    else:
        vectorizer = CountVectorizer(stop_words='english')
        count_data = vectorizer.fit_transform(data)
        return count_data

def apply_unsupervised_learning(data, file_type):
    """Apply an unsupervised learning technique based on the type of data."""
    if file_type == 'csv':
        kmeans = KMeans(n_clusters=5)  # Assume 3 clusters for simplicity
        labels = kmeans.fit_predict(data)
        return {'cluster labels': labels.tolist()}
    else:
        lda = LatentDirichletAllocation(n_components=5)  # Assume 5 topics
        topic_distribution = lda.fit_transform(data)
        return {'topic_distribution': topic_distribution.tolist()}

def execute_full_unsupervised_learning(file_stream, file_type):
    """ Execute all unsupervised learning steps in sequence. """
    data = load_data(file_stream)
    processed_data = preprocess_data(data, file_type)
    results = apply_unsupervised_learning(processed_data, file_type)
    return results