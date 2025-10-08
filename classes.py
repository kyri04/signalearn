import numpy as np

class Series:
        
    def __init__(self, parameters):

        self.name = parameters['name']
        self.xlabel = parameters['xlabel']
        self.ylabel = parameters['ylabel']
        self.xunit = parameters['xunit']
        self.yunit = parameters['yunit']
        self.x = parameters['x']
        self.y = parameters['y']

        self.title = self.name
        self.filename = self.name

class ClassificationResult:

    def __init__(self, set_params, set_results, set_group_results=None):
        self.params = set_params
        self.results = set_results
        self.group_results = set_group_results

    def format_params(self):
        lines = ["PARAMETERS:"]
        for k, v in vars(self.params).items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def format_results(self, results, title="RESULTS:"):
        def fmt(x, pct=False):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "N/A"
            return f"{x * 100:.2f}%" if pct else f"{x:.2f}"

        return (
            f"{title}\n"
            f"Accuracy: {fmt(results.accuracy, True)}\n"
            f"Precision: {fmt(results.precision, True)}\n"
            f"Recall: {fmt(results.recall, True)}\n"
            f"Specificity: {fmt(results.specificity, True)}\n"
            f"Sensitivity: {fmt(results.sensitivity, True)}\n"
            f"F1: {fmt(results.f1)}\n\n"
            f"{self.display_confusion_matrix(results.conf_matrix)}\n\n"
        )
    
    def display_confusion_matrix(self, conf_matrix):
        col_width = max(5, max(len(str(label)) for label in self.params.unique_labels))
        row_header_width = max(len("Actual " + str(label)) for label in self.params.unique_labels)
        output = []
        header_width = len(self.params.unique_labels) * (col_width + 1) - 1
        output.append(" " * (row_header_width + 1) + "Predicted".center(header_width))
        output.append(" " * (row_header_width + 1) + " ".join(f"{label:^{col_width}}" for label in self.params.unique_labels))
        for i, label in enumerate(self.params.unique_labels):
            row_header = f"Actual {label}"
            row_values = " ".join(f"{conf_matrix[i, j]:^{col_width}}" for j in range(len(self.params.unique_labels)))
            output.append(f"{row_header:<{row_header_width}} " + row_values)
        return "\n".join(output)
    
    def display_params(self): print(self.format_params())
    def display_results(self): print(self.format_results(self.results, title="RESULTS:"))
    def display_group_results(self): print(self.format_results(self.group_results, title="GROUP-LEVEL RESULTS:"))