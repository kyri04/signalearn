from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import gc

from utility import *
from signalearn.classes import ClassificationResult
from signalearn.utility import *

def calculate_folds(length):
    return min(10, max(3, round((length / 100) ** 0.5)))

def reduce(y, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(y)

def scale(y):
    scaler = StandardScaler()
    return np.array(scaler.fit_transform(y))

def encode(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return np.array(encoded_labels), encoder

def get_classifier(classify_method):
    classifiers = {
        'dt': DecisionTreeClassifier(random_state=42),
        'rf': RandomForestClassifier(random_state=42, n_estimators=300, n_jobs=-1),
        'svm': SVC(random_state=42, probability=True),
        'lr': LogisticRegression(random_state=42),
        'knn': KNeighborsClassifier(),
        'gb': GradientBoostingClassifier(random_state=42)
    }
    return classifiers[classify_method]

def get_param_grid(classify_method):
    searchers = {
        'dt': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
        },
        'rf': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'criterion': ['gini', 'entropy'],
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'],
        },
        'lr': {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 500],
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
        },
        'gb': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
        },
    }
    return searchers[classify_method]

def calculate_metrics(conf_matrix):
    specificity = []
    sensitivity = []
    precision = []
    recall = []
    
    total_tp = np.trace(conf_matrix)
    total_samples = conf_matrix.sum()

    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]
        fn = conf_matrix[i, :].sum() - tp
        fp = conf_matrix[:, i].sum() - tp
        tn = total_samples - (tp + fn + fp)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else None
        sensitivity.append(sens)
        recall.append(sens)

        spec = tn / (tn + fp) if (tn + fp) > 0 else None
        specificity.append(spec)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        precision.append(prec)

    accuracy = total_tp / total_samples if total_samples > 0 else None

    mean_sensitivity = np.mean([val for val in sensitivity if val is not None]) if sensitivity else None
    mean_specificity = np.mean([val for val in specificity if val is not None]) if specificity else None
    mean_precision = np.mean([val for val in precision if val is not None]) if precision else None
    mean_recall = np.mean([val for val in recall if val is not None]) if recall else None

    return accuracy, mean_specificity, mean_sensitivity, mean_precision, mean_recall

def display_confusion_matrix(conf_matrix, labels):
    header_width = len(labels) * 6 - 1
    output = []
    
    output.append(" " * 10 + "Predicted".center(header_width))
    
    output.append(" " * 10 + " ".join(f"{label:^5}" for label in labels))
    
    for i, label in enumerate(labels):
        row = f"{'Actual ' + str(label):<10}" + " ".join(f"{conf_matrix[i, j]:^5}" for j in range(len(labels)))
        output.append(row)
    
    return "\n".join(output)

def sum_confusion_matrices(conf_matrices):
    overall_conf_matrix = np.sum(conf_matrices, axis=0)
    return overall_conf_matrix

def classify(
        points, 
        label, 
        group = None, 
        classifier = 'rf', 
        display_results=True, 
        save_results=True,
        cross_validate=False, 
        tune=False
    ):

    point_type = points[0].__class__.__name__.lower()

    result_string = (

        (f"TYPE: {point_type}") +
        (f"LABELS: {label}\n" if label is not None else '') +
        (f"GROUP: {group}\n" if group is not None else '') +
        (f"CLASSIFIER: {classifier}\n" if classifier is not None else '') + '\n\n'

    )
    print(result_string)

    N = len(points)
    
    ys = [point.y for point in points]
    xs = [point.x for point in points]
    
    labels = np.array([getattr(point, label) for point in points], dtype=str)
    groups = np.array([getattr(point, group) for point in points], dtype=str) if group is not None else None

    ys_scaled = scale(ys)
    labels_encoded, encoder = encode(labels)

    unique_labels = encoder.classes_
    unique_labels_encoded = np.unique(labels_encoded)

    attr_same, val_same = find_same_attribute(points)
    name = f"{point_type}-{label}{'-'+group if group is not None else ''}-{classifier}{'-crossval' if cross_validate else ''}{'-tuning' if tune else ''}{('-'+attr_same+'='+val_same) if attr_same is not None and val_same is not None else ''}"
    result = ClassificationResult(name, unique_labels, xs[0])

    nfolds = 1
    if cross_validate:
        cv_strategy = GroupKFold(n_splits=len(np.unique(groups))) if group is not None else KFold(n_splits=calculate_folds(N))
        splits = cv_strategy.split(ys_scaled, labels_encoded, groups)
        nfolds = cv_strategy.n_splits
    else:
        test_size = 1 / len(np.unique(groups)) if group is not None else 0.2
        train_idx, test_idx = train_test_split(
            range(N), test_size=test_size, stratify=labels_encoded if group is None else None, random_state=42
        )
        splits = [(train_idx, test_idx)]

    conf_matrices = []

    count = 1
    for train_idx, test_idx in splits:
        
        X_train, X_test = ys_scaled[train_idx], ys_scaled[test_idx]
        y_train, y_test = labels_encoded[train_idx], labels_encoded[test_idx]

        labels_same = np.all(y_test == y_test[0])

        model = get_classifier(classifier)
        param_grid = get_param_grid(classifier)

        if tune:
            search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, scoring='accuracy', cv=3, random_state=42, n_jobs=-1, verbose=1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
        else:
            model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)

        conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels_encoded)
        class_report = classification_report(y_test, y_pred, labels=unique_labels_encoded, zero_division=np.nan) if not labels_same else None

        accuracy, specificity, sensitivity, precision, recall = calculate_metrics(conf_matrix)

        conf_matrices.append(conf_matrix)
        
        result.add_model(
            model, 
            X_test, 
            y_test,
            accuracy,
            np.array(points)[test_idx]
        )

        mean_text = "Mean " if conf_matrix.shape[0] > 2 else ""
        display = (

            f"({count}/{nfolds})" + '\n' +
            (f"GROUP: {groups[test_idx][0]}" + f" with labels {np.unique(labels[test_idx])}\n" if group is not None else '') +
            f"Accuracy: {accuracy * 100:.2f}%\n" +
            (f"{mean_text}Precision: {precision * 100:.2f}%\n" if not labels_same else '') +
            (f"{mean_text}Recall: {recall * 100:.2f}%\n" if not labels_same else '') +
            (f"{mean_text}Specificity: {specificity * 100:.2f}%\n" if not labels_same else '') +
            (f"{mean_text}Sensitivity: {sensitivity * 100:.2f}%\n" if not labels_same else '') + "\n" +
            display_confusion_matrix(conf_matrix, unique_labels) +
            "\n\n-------------------------------------\n\n"

        )
        if nfolds > 1: result_string += display

        if display_results and cross_validate and nfolds > 1:
            print(display)

        count += 1

        del X_train, X_test, y_train, y_test, y_pred, model
        del accuracy, precision, recall, specificity, sensitivity, class_report, conf_matrix
        gc.collect()

    conf_matrix_sum = sum_confusion_matrices(conf_matrices)
    accuracy, specificity, sensitivity, precision, recall = calculate_metrics(conf_matrix_sum)

    display = (

        ("OVERALL RESULTS\n" if nfolds > 1 or cross_validate else 'RESULTS\n') + 
        f"Accuracy: {accuracy * 100:.2f}%\n" +
        (f"{mean_text}Precision: {precision * 100:.2f}%\n" if not labels_same else '') +
        (f"{mean_text}Recall: {recall * 100:.2f}%\n" if not labels_same else '') +
        (f"{mean_text}Specificity: {specificity * 100:.2f}%\n" if not labels_same else '') +
        (f"{mean_text}Sensitivity: {sensitivity * 100:.2f}%\n" if not labels_same else '') + "\n" +
        display_confusion_matrix(conf_matrix_sum, unique_labels)

    )
    result_string += display

    if save_results:

        os.makedirs('results', exist_ok=True)
        with open(f"results/results-{name}.txt", "w") as file:
            file.write(result_string)

    if display_results:
        print(display)

    return result