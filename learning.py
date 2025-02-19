from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold, KFold, train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import gc

from signalearn.classes import ClassificationResult
from signalearn.utility import *
from signalearn.general_utility import time

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
        'rf': RandomForestClassifier(random_state=42, n_estimators=300, max_depth=10, n_jobs=-1),
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
            'n_estimators': [300],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None, 0.2, 0.5, 1.0], 
            'bootstrap': [True, False],
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
    f1_scores = []  # List to store F1 scores for each class
    
    total_tp = np.trace(conf_matrix)
    total_samples = conf_matrix.sum()

    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]
        fn = conf_matrix[i, :].sum() - tp
        fp = conf_matrix[:, i].sum() - tp
        tn = total_samples - (tp + fn + fp)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else None
        sensitivity.append(sens)
        recall.append(sens)  # recall is the same as sensitivity

        spec = tn / (tn + fp) if (tn + fp) > 0 else None
        specificity.append(spec)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        precision.append(prec)
        
        # Calculate F1 score if both precision and sensitivity are available
        if prec is not None and sens is not None and (prec + sens) > 0:
            f1 = 2 * (prec * sens) / (prec + sens)
        else:
            f1 = None
        f1_scores.append(f1)

    accuracy = total_tp / total_samples if total_samples > 0 else None

    mean_sensitivity = np.mean([val for val in sensitivity if val is not None]) if sensitivity else None
    mean_specificity = np.mean([val for val in specificity if val is not None]) if specificity else None
    mean_precision = np.mean([val for val in precision if val is not None]) if precision else None
    mean_recall = np.mean([val for val in recall if val is not None]) if recall else None
    mean_f1 = np.mean([val for val in f1_scores if val is not None]) if f1_scores else None

    return accuracy, mean_specificity, mean_sensitivity, mean_precision, mean_recall, mean_f1

def display_confusion_matrix(conf_matrix, labels):
    # Determine the minimum column width.
    # This is the max between the default width (5) and the length of the longest label.
    col_width = max(5, max(len(str(label)) for label in labels))
    
    # For row headers we display "Actual " + label.
    # Compute the maximum length of these row header strings.
    row_header_width = max(len("Actual " + str(label)) for label in labels)
    
    output = []
    
    # Total width for the header (each column plus a separating space)
    header_width = len(labels) * (col_width + 1) - 1
    
    # First line: "Predicted" centered, with some left padding for the row headers.
    output.append(" " * (row_header_width + 1) + "Predicted".center(header_width))
    
    # Second line: predicted labels with proper spacing.
    output.append(" " * (row_header_width + 1) +
                  " ".join(f"{label:^{col_width}}" for label in labels))
    
    # Subsequent lines: each row with its row header and the matrix values.
    for i, label in enumerate(labels):
        # Create a row header like "Actual X"
        row_header = f"Actual {label}"
        # Format each element in the row to have the same width.
        row_values = " ".join(f"{conf_matrix[i, j]:^{col_width}}" 
                              for j in range(len(labels)))
        # Left-align the row header within its width and add the values.
        output.append(f"{row_header:<{row_header_width}} " + row_values)
    
    return "\n".join(output)

def sum_confusion_matrices(conf_matrices):
    overall_conf_matrix = np.sum(conf_matrices, axis=0)
    return overall_conf_matrix
from imblearn.over_sampling import RandomOverSampler
def classify(
        points, 
        label, 
        group=None, 
        classifier='rf', 
        display_results=True, 
        save_results=True,
        cross_validate=False, 
        test_size=0.2,
        tune=False,
        oversample=False
    ):

    # Print initial configuration
    point_type = points[0].__class__.__name__.lower()
    result_string = (
        f"TYPE: {point_type}\n" +
        (f"LABELS: {label}\n" if label is not None else '') +
        (f"GROUP: {group}\n" if group is not None else '') +
        (f"CLASSIFIER: {classifier}\n" if classifier is not None else '') + '\n\n'
    )
    print(result_string)
    
    N = len(points)
    ys = np.array([point.y for point in points])
    xs = np.array([point.x for point in points])
    
    # Prepare labels based on input type
    if isinstance(label, str):
        labels = np.array([getattr(point, label) for point in points], dtype=str)
    elif isinstance(label, list):
        labels = np.array(["_".join(str(getattr(point, attr)) for attr in label) for point in points], dtype=str)
    else:
        raise ValueError("label must be either a string or a list")
    
    groups = np.array([getattr(point, group) for point in points], dtype=str) if group is not None else None

    # Scale and encode labels
    ys_scaled = scale(ys)
    labels_encoded, encoder = encode(labels)
    unique_labels = encoder.classes_
    unique_labels_encoded = np.unique(labels_encoded)
    
    attr_same, val_same = find_same_attribute(points)
    name = (f"{point_type}-{label}{'-' + group if group is not None else ''}-"
            f"{classifier}{'-crossval' if cross_validate else ''}"
            f"{'-tuning' if tune else ''}"
            f"{('-' + attr_same + '=' + val_same) if attr_same is not None and val_same is not None else ''}")
    result = ClassificationResult(name, unique_labels, xs[0])
    
    if cross_validate:
        if group is not None:
            cv_strategy = GroupKFold(n_splits=len(np.unique(groups)))
            splits = cv_strategy.split(ys_scaled, labels_encoded, groups)
            nfolds = cv_strategy.n_splits

            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert train_groups.isdisjoint(test_groups), "Data from the same group found in both training and testing sets!"
        else:
            nfolds = calculate_folds(N)
            cv_strategy = KFold(n_splits=nfolds)
            splits = cv_strategy.split(ys_scaled, labels_encoded)
    else:
        # Use a 60/40 split
        if group is not None:
            gss = GroupShuffleSplit(test_size=test_size, random_state=42)
            train_idx, test_idx = next(gss.split(np.arange(N), labels_encoded, groups))

            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert train_groups.isdisjoint(test_groups), "Data from the same group found in both training and testing sets!"
        else:
            train_idx, test_idx = train_test_split(
                range(N), test_size=test_size, stratify=labels_encoded, random_state=42
            )
        splits = [(train_idx, test_idx)]
        nfolds = 1

    conf_matrices = []
    count = 1
    for train_idx, test_idx in splits:
        X_train, X_test = ys_scaled[train_idx], ys_scaled[test_idx]
        y_train, y_test = labels_encoded[train_idx], labels_encoded[test_idx]
        labels_same = np.all(y_test == y_test[0])

        if oversample:
            oversampler = RandomOverSampler(random_state=42)
            X_train, y_train = oversampler.fit_resample(X_train, y_train)

        model = get_classifier(classifier)
        param_grid = get_param_grid(classifier)
    
        if tune:
            search = RandomizedSearchCV(
                model, param_distributions=param_grid, n_iter=20, scoring='accuracy',
                cv=3, random_state=42, n_jobs=-1, verbose=1
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print(search.best_params_)
        else:
            model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels_encoded)
        class_report = (classification_report(y_test, y_pred, labels=unique_labels_encoded, zero_division=np.nan)
                        if not labels_same else None)
        accuracy, specificity, sensitivity, precision, recall, f1 = calculate_metrics(conf_matrix)
        conf_matrices.append(conf_matrix)
    
        result.add_model(model, X_test, y_test, f1, np.array(points)[test_idx])
    
        mean_text = "Mean " if conf_matrix.shape[0] > 2 else ""
        split_display = (
            f"({count}/{nfolds})\n" +
            (f"GROUP: {groups[test_idx][0]} with labels {np.unique(labels[test_idx])}\n" if group is not None else '') +
            f"Accuracy: {accuracy * 100:.2f}%\n" +
            (f"{mean_text}Precision: {precision * 100:.2f}%\n" if not labels_same else '') +
            (f"{mean_text}Recall: {recall * 100:.2f}%\n" if not labels_same else '') +
            (f"{mean_text}Specificity: {specificity * 100:.2f}%\n" if not labels_same else '') +
            (f"{mean_text}Sensitivity: {sensitivity * 100:.2f}%\n" if not labels_same else '') +
            (f"{mean_text}F1: {f1:.2f}%\n" if not labels_same else '') +
            "\n" +
            display_confusion_matrix(conf_matrix, unique_labels) +
            "\n\n-------------------------------------\n\n"
        )
        if nfolds > 1:
            result_string += split_display
        if display_results and cross_validate and nfolds > 1:
            print(split_display)
    
        count += 1
        del X_train, X_test, y_train, y_test, y_pred, model
        gc.collect()
    
    conf_matrix_sum = sum_confusion_matrices(conf_matrices)
    overall_accuracy, overall_specificity, overall_sensitivity, overall_precision, overall_recall, overall_f1 = calculate_metrics(conf_matrix_sum)
    overall_display = (
        ("OVERALL RESULTS\n" if nfolds > 1 or cross_validate else 'RESULTS\n') +
        f"Accuracy: {overall_accuracy * 100:.2f}%\n" +
        (f"{mean_text}Precision: {overall_precision * 100:.2f}%\n" if not labels_same else '') +
        (f"{mean_text}Recall: {overall_recall * 100:.2f}%\n" if not labels_same else '') +
        (f"{mean_text}Specificity: {overall_specificity * 100:.2f}%\n" if not labels_same else '') +
        (f"{mean_text}Sensitivity: {overall_sensitivity * 100:.2f}%\n" if not labels_same else '') +
        (f"{mean_text}F1: {overall_f1:.2f}\n" if not labels_same else '') +
        "\n" +
        display_confusion_matrix(conf_matrix_sum, unique_labels)
    )
    result_string += overall_display
    
    if save_results:
        os.makedirs('results', exist_ok=True)
        with open(f"results/results-{name}_{time()}.txt", "w") as file:
            file.write(result_string)
    if display_results:
        print(overall_display)
    
    return result

import matplotlib.pyplot as plt
import random

def performance_evolution(points, attr, steps=10):
    """
    Trains models with an increasing number of groups (based on the specified attribute)
    and plots the resulting F1 score evolution.
    
    For example, if attr is "id_patient" and there are 200 unique patients, the function
    will first train a model using points from the first 10 patients, then 20 patients,
    and so on until all 200 patients are used.
    
    Args:
        points (list): List of point objects.
        attr (str): The attribute name used to group points (e.g. "id_patient").
        steps (int): The increment in the number of groups for each evaluation (e.g. 10 patients per step).
    
    Returns:
        tuple: Two lists containing the number of groups used and the corresponding F1 scores.
    """
    # Get unique group identifiers from the points using the specified attribute.
    unique_groups = sorted(set(getattr(point, attr) for point in points))
    total_groups = len(unique_groups)
    
    # If the step size is larger than the number of groups, adjust it.
    if steps > total_groups:
        steps = total_groups

    # Create a list of group counts to evaluate.
    group_counts = list(range(steps, total_groups + 1, steps))
    # Ensure that the final step uses all groups.
    if group_counts[-1] != total_groups:
        group_counts.append(total_groups)
    
    f1_scores = []
    
    # For each group count, filter the points, train a model, and record the F1 score.
    for num_groups in group_counts:
        # Select the first num_groups from the sorted unique groups.
        selected_groups = set(unique_groups[:num_groups])
        # Filter points that belong to the selected groups.
        subset = [p for p in points if getattr(p, attr) in selected_groups]
        
        # Train a model using the classify function (with display and saving disabled).
        result = classify(subset, label='diagnosis', group='id_patient', display_results=False, save_results=False, cross_validate=False, tune=False)
        
        # Extract the F1 score from the classification result.
        # When cross_validate is False, result.scores should contain a single value.
        f1 = result.scores[0] if result.scores else None
        f1_scores.append(f1)
    
    # Plot the performance evolution.
    plt.figure()
    plt.plot(group_counts, f1_scores, marker='o', linestyle='-')
    plt.xlabel(f'Number of {attr} groups')
    plt.ylabel('F1 Score')
    plt.title(f'Performance Evolution by Increasing Number of {attr} Groups')
    plt.grid(True)
    plt.show()
    
    return group_counts, f1_scores