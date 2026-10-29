# Predictions
Use `sl.predict(x, model)` for native class labels or regression values. The
optional `output` selects a scientifically different result where supported:

| Output | Name | Structure |
| --- | --- | --- |
| `"probabilities"` | `<target>_probability` | One class vector per sample. |
| `"scores"`, binary | `<target>_score` | Signed margin. |
| `"scores"`, multiclass | `<target>_score` | Per-class score vector. |
| `"logits"` | `<target>_logit` | Per-class logit vector. |

Sklearn exposes probabilities or scores when its native methods support them.
[sl.Model](models.md) exposes probabilities and explicitly declared logits.
Decision scores are not inferred from logits.

An external/unnamed/id target uses the base `prediction`. IDs and current sample
order are preserved. Probability, score, and logit values are finite non-nullable
float64 and do not inherit target units. Series colour/marker may be retained; category
display belongs to the class representation. Class order follows sklearn
`classes_` or the explicit `sl.Model.classes` declaration, preserving label types.

For example, with a fitted digit classifier:

```python
probabilities = sl.predict(test_features, model, output="probabilities")
probabilities.digit_probability.coords["class"]
probabilities[0].digit_probability
```

The whole Field has shape `(samples, classes)` and dims `("class",)`. One
sample's value has shape `(classes,)`. Binary scores retain both classes and
`positive_class=classes_[1]`, matching the native signed-margin direction.
Positive values point towards that class. Scores are labelled `decision_scores`,
not probabilities or logits.

## Decisions
Sklearn preserves native label decisions, even when they differ from probability
argmax. `sl.Model` uses argmax over its declared classes, choosing the first
class in a tie. Training encodes these classes to indices; prediction restores
the original labels.

For `sl.Model(output="logits")`, requesting probabilities applies softmax.
Declared probabilities are validated without correction. Regression returns
finite float64 values with target units. Class-label predictions retain target
display metadata; probabilities, scores, and logits do not inherit target units.

## Constraints
Check output availability before consuming data. Native multilabel/multioutput
tasks and multiclass one-vs-one pair scores are rejected. `sl.Model` also rejects
scalar binary logits and structured classification outputs. Pairwise columns
need a separate class-pair representation. Invalid returned shape, sample count, missing/masked/infinite values, probabilities
outside [0,1], or incompatible classes error when consumed. Exclusive vectors
must sum to one within `rtol=1e-6, atol=1e-8`. Declared probabilities are not
clipped, renormalised, or calibrated.

Prediction metadata is read-only and survives selection, assignment,
concatenation, cross-validation, and save/load. Equivalent class columns align
by label to first-input order during concatenation. Different vocabularies,
missing fold classes, incompatible meaning/display metadata, or reversed scalar
score direction are errors. No missing probabilities are fabricated. Slicing,
cropping, or windowing the class axis is rejected; other compatible axes remain
available for explicitly structured custom outputs.

Existing label metrics and confusion plots reject these outputs. ROC curves
and probability-based metrics are not implemented. The general representation
below also supports explicitly declared logits/multilabel outputs; the current
sklearn adapter does not provide those native prediction modes.

## Schema
Probability, decision-score, and logit fields can declare a `prediction` object.
It is optional for ordinary measurements and class-label fields. It records
scientific meaning independently of the model library:

```json
{
  "dtype": "float64",
  "dims": ["class"],
  "shape": [3],
  "coords": {"class": [2, 7, 11]},
  "prediction": {
    "kind": "probabilities",
    "task": "multiclass",
    "target": "digit",
    "classes": [2, 7, 11],
    "class_metadata": {
      "label": "Digit",
      "label_latex": "$d$",
      "colours": [[2, "red"], [7, "blue"], [11, "green"]],
      "markers": []
    }
  }
}
```

An array field also needs the usual `source` declaration. A scalar binary score
uses the scalar schema instead, without `dims`, `shape`, or `coords`.

`kind`, `task`, `target`, and `classes` are required. `kind` is `probabilities`,
`decision_scores`, or `logits`; no conversion between them is inferred. `task`
is `binary`, `multiclass`, or `multilabel`. Classes are unique, homogeneous
scalar labels, preserved in order and type. Binary outputs have exactly two
classes. Per-class outputs need a `class` dimension with matching inline
coordinates. Other dimensions can represent time, space, or other explicitly
declared structure.

A binary score or logit may omit the class dimension and instead declare
`positive_class`, identifying which class positive values favour. Per-class
outputs do not take `positive_class`. Prediction values are floating-point and
non-nullable. Probabilities lie in `[0, 1]`; exclusive-class vectors sum to one
within `rtol=1e-6, atol=1e-8`. Multilabel probabilities are independent and have
no sum-to-one requirement. Scores and logits may be negative or unnormalised;
all values must be finite. Validation does not clip or normalise values.

Optional `class_metadata` contains `label`, `label_latex`, `colours`, and
`markers`, describing the class coordinate. In JSON the two mappings use lists
of `[category, display]` pairs, retaining typed category keys. In the Python
`Field(..., prediction=...)` constructor they use ordinary mappings. Category
displays attach to classes, rather than numeric probability values. The
read-only `field.prediction` property exposes the validated representation.

This representation can describe explicitly declared custom component outputs.
It does not imply that all tasks have native learning integrations. The current
sklearn prediction adapter supports single-target binary and exclusive
multiclass probabilities and decision scores; it does not support native
multilabel, logit, or pairwise-score dispatch. See the
[learning reference](learning.md#predict).
