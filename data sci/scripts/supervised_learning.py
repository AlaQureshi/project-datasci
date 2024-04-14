import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data, None
    except Exception as e:
        return None, str(e)

def preprocess_data(df, target, continuous=True):
    """ Preprocess data by handling missing values and splitting into features and target. """
    df = df.dropna()
    X = df.drop(columns=[target])
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, model_type='linear_regression'):
    """ Train a model based on the specified type. """
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'knn_regression':
        model = KNeighborsRegressor()
    elif model_type == 'svr':
        model = SVR()
    elif model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'perceptron':
        model = Perceptron()
    elif model_type == 'svm':
        model = SVC()
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    elif model_type == 'knn_classifier':
        model = KNeighborsClassifier()
    elif model_type == 'naive_bayes':
        model = GaussianNB()
    else:
        raise ValueError("Invalid model type provided.")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, continuous=True):
    """ Evaluate the model using the appropriate metric. """
    y_pred = model.predict(X_test)
    if continuous:
        mse = mean_squared_error(y_test, y_pred)
        return {'Mean Squared Error': mse}
    else:
        accuracy = accuracy_score(y_test, y_pred)
        return {'Accuracy': accuracy}

def execute_full_supervised_learning(file_path, target_column, model_type='logistic_regression'):
    """Execute all supervised learning steps in sequence."""
    data = load_data(file_path)
    X, y = preprocess_data(data, target_column)
    model, X_test, y_test = train_model(X, y, model_type)
    evaluation_results = evaluate_model(model, X_test, y_test)
    return evaluation_results
