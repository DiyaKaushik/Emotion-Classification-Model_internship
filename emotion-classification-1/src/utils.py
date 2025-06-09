def calculate_accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

def calculate_f1_score(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='weighted')

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    import joblib
    return joblib.load(filename)