class EmotionClassifier:
    def __init__(self, model=None):
        self.model = model

    def train(self, X_train, y_train):
        if self.model is None:
            raise ValueError("Model is not defined.")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not defined.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model is not defined.")
        predictions = self.predict(X_test)
        accuracy = (predictions == y_test).mean()
        return accuracy

def create_model(model_type='logistic'):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    if model_type == 'logistic':
        return LogisticRegression()
    elif model_type == 'random_forest':
        return RandomForestClassifier()
    elif model_type == 'svm':
        return SVC()
    else:
        raise ValueError("Unsupported model type. Choose 'logistic', 'random_forest', or 'svm'.")