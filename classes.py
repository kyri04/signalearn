from signalearn.general_utility import *
from signalearn.utility import *

class Series:
    def __init__(self, parameters):
        for k, v in parameters.items():
            setattr(self, k, v)

class Result:
    def __init__(self, set_params, set_volume, set_results, set_meta):
        self.params = make_namespace(set_params)
        self.volume = make_namespace(set_volume)
        self.results = make_namespace(set_results)
        self.meta = make_namespace(set_meta)

        raw_cm = getattr(self.meta, "conf_matrix", None)
        labels = getattr(self.params, "unique_labels", None)

        if raw_cm is not None and labels is not None:
            self.results.conf_matrix = format_confusion_matrix(raw_cm, labels)
    
    def display_params(self): print(format_attributes(self.params, 'PARAMETERS:'))
    def display_volume(self): print(format_attributes(self.volume, 'VOLUME:'))
    def display_results(self): print(format_attributes(self.results, 'RESULTS:'))
    def display(self):
        self.display_params()
        self.display_volume()
        self.display_results()