import numpy as np
from numbers import Number

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

class Result:

    def __init__(self, set_params, set_volume, set_results, set_group_results=None):
        self.params = set_params
        self.volume = set_volume
        self.results = set_results
        self.group_results = set_group_results

    def format_params(self):
        lines = ["PARAMETERS:"]
        for k, v in vars(self.params).items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)
    
    def format_volume(self):
        lines = ["VOLUME:"]
        for k, v in vars(self.volume).items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def format_results(self, results, title="RESULTS:"):

        def to_float(x):
            if x is None:
                return None
            try:
                if isinstance(x, np.ndarray):
                    if x.size == 0:
                        return None
                    x = x.ravel()[0]
                if not isinstance(x, Number):
                    x = float(x)
                if isinstance(x, float) and np.isnan(x):
                    return None
                return float(x)
            except Exception:
                return None

        def pct(x):  v = to_float(x); return "N/A" if v is None else f"{v*100:.2f}%"
        def num(x):  v = to_float(x); return "N/A" if v is None else f"{v:.2f}"

        lines = [
            f"{title}",
            f"Accuracy: {pct(getattr(results, 'accuracy', None))}",
            f"Precision: {pct(getattr(results, 'precision', None))}",
            f"Recall: {pct(getattr(results, 'recall', None))}",
            f"Specificity: {pct(getattr(results, 'specificity', None))}",
            f"Sensitivity: {pct(getattr(results, 'sensitivity', None))}",
            f"F1: {num(getattr(results, 'f1', None))}",
            "",
            self.display_confusion_matrix(getattr(results, 'conf_matrix', None)) or "",
            ""
        ]
        return "\n".join(lines)
    
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
    def display_volume(self): print(self.format_volume())
    def display_results(self): print(self.format_results(self.results, title="RESULTS:"))
    def display_group_results(self): print(self.format_results(self.group_results, title="GROUP-LEVEL RESULTS:"))
    def display(self):
        self.display_params()
        print()
        self.display_volume()
        print()
        self.display_results()
        if self.group_results:
            print()
            self.display_group_results()