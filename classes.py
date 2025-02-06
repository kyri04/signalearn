class ClassificationResult:

    def __init__(self, set_name, set_classes, set_x_range):
        self.name = set_name
        self.feature_importances = []
        self.class_probabilities = []
        self.labels = []
        self.scores = []
        self.points = []
        self.classes = set_classes
        self.x_range = set_x_range

    def add_model(self, model, X_test, y_test, score, points):

        if hasattr(model, 'feature_importances_'): self.feature_importances.append(getattr(model, 'feature_importances_'))
        else: self.feature_importances.append(None)

        self.scores.append(score)

        self.class_probabilities.append(model.predict_proba(X_test))
        self.labels.extend(y_test)
        self.points.extend(points)