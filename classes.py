from signalearn.general_utility import *
from signalearn.utility import *

class Field:
    def __init__(self, values, label=None, unit=None, name=None):
        self.values = values
        self.label = label
        self.unit = unit
        self.name = name

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype) if dtype is not None else np.asarray(self.values)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]

    def __getattr__(self, name):
        return getattr(self.values, name)

class FieldCollection:
    def __init__(self, fields, name=None, dataset=None):
        self.fields = list(fields)
        self._name = name
        self._dataset = dataset

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        old = self._name
        self._name = value
        for f in self.fields:
            f.name = value
        if self._dataset is None or old is None or old == value:
            return
        for s in self._dataset.samples:
            if old in s.fields:
                s.fields[value] = s.fields.pop(old)

    @property
    def label(self):
        if not self.fields:
            return None
        lbl = self.fields[0].label
        for f in self.fields[1:]:
            if f.label != lbl:
                return None
        return lbl

    @label.setter
    def label(self, value):
        for f in self.fields:
            f.label = value

    @property
    def unit(self):
        if not self.fields:
            return None
        unit = self.fields[0].unit
        for f in self.fields[1:]:
            if f.unit != unit:
                return None
        return unit

    @unit.setter
    def unit(self, value):
        for f in self.fields:
            f.unit = value

    def __array__(self, dtype=None):
        arr = [np.asarray(f) for f in self.fields]
        return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, idx):
        return self.fields[idx]

class Sample:
    def __init__(self, parameters=None, **kwargs):
        object.__setattr__(self, "_fields", {})
        params = {}
        if parameters is not None:
            params.update(parameters)
        if kwargs:
            params.update(kwargs)

        labels = params.pop("labels", None)
        units = params.pop("units", None)

        for k, v in params.items():
            label = labels.get(k) if isinstance(labels, dict) else None
            unit = units.get(k) if isinstance(units, dict) else None
            if isinstance(v, Field):
                if v.name is None:
                    v.name = k
                self._fields[k] = v
            else:
                self._fields[k] = Field(v, label=label, unit=unit, name=k)

    @property
    def fields(self):
        return self._fields

    def __getattr__(self, name):
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        fields = object.__getattribute__(self, "_fields")
        if isinstance(value, Field):
            if value.name is None:
                value.name = name
            fields[name] = value
        else:
            fields[name] = Field(value, name=name)

    def __dir__(self):
        names = set(object.__dir__(self))
        names.update(self._fields.keys())
        return sorted(names)

class Dataset:
    def __init__(self, samples=None):
        self.samples=samples
        
    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Dataset(self.samples[idx])
        return self.samples[idx]

    def __getattr__(self, name):
        fields = [s.fields[name] for s in self.samples if name in s.fields]
        if not fields:
            raise AttributeError(name)
        return FieldCollection(fields, name=name, dataset=self)

    def __dir__(self):
        names = set(object.__dir__(self))
        for s in self.samples:
            names.update(s.fields.keys())
        return sorted(names)

class Result:
    def __init__(self, set_params, set_results, set_meta):
        self.params = make_namespace(set_params)
        self.results = make_namespace(set_results)
        self.meta = make_namespace(set_meta)

        raw_cm = getattr(self.meta, "conf_matrix", None)
        labels = getattr(self.params, "unique_labels", None)

        if raw_cm is not None and labels is not None:
            self.results.conf_matrix = format_confusion_matrix(raw_cm, labels)
    
    def display_params(self): print(format_attributes(self.params, 'PARAMETERS:'))
    def display_results(self): print(format_attributes(self.results, 'RESULTS:'))
    def display(self):
        self.display_params()
        self.display_results()
