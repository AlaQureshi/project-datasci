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
        return data.dropna()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def preprocess_data(data, target_column, scale_features=False):
    if data is None:
        print("No data provided.")
        return None, None
    y = data[target_column]
    X = data.drop(columns=[target_column])
    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, model_type):
    model_dict = {
        'linear_regression': LinearRegression(),
        'knn_regression': KNeighborsRegressor(),
        'svr': SVR(),
        'logistic_regression': LogisticRegression(),
        'perceptron': Perceptron(),
        'svm': SVC(),
        'decision_tree': DecisionTreeClassifier(),
        'knn_classifier': KNeighborsClassifier(),
        'naive_bayes': GaussianNB()
    }
    model = model_dict.get(model_type)
    if model is None:
        raise ValueError("Invalid model type provided.")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, is_continuous=True):
    y_pred = model.predict(X_test)
    if is_continuous:
        return {'Mean Squared Error': mean_squared_error(y_test, y_pred)}
    else:
        return {'Accuracy': accuracy_score(y_test, y_pred)}

def execute_full_supervised_learning(filepath, target_column, model_type, is_continuous=True):
    data = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column, scale_features=not is_continuous)
    model = train_model(X_train, y_train, model_type)
    results = evaluate_model(model, X_test, y_test, is_continuous)
    return results

