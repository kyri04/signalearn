# Evaluation
Metrics take predictions first and the true target field second. SignaLearn
aligns them by sample ID. Both must describe the same complete set of observations.

For class predictions in `predictions.state` and targets in `test.state`:

```python
accuracy = sl.accuracy(predictions.state, test.state)
macro_f1 = sl.f1(predictions.state, test.state, average="macro")
per_class = sl.f1(predictions.state, test.state, average=None)
```

## Metrics
| Task | Functions |
| --- | --- |
| Classification | `accuracy`, `precision`, `recall`, `specificity`, `f1` |
| Regression | `mae`, `rmse`, `r2` |

`sensitivity` is an alias for `recall`. Precision, recall, specificity, and F1
require `average`: use `macro` for equal class weight, `weighted` for support
weighting, `micro` to pool counts, or `None` for per-class results. Binary
scores require `average="binary"` and an explicit `positive` class.

Undefined scores warn and remain distinguishable from zero. Metrics do not
drop missing values, threshold probabilities, or convert units.

## Groups
Pass a grouping field to score each group separately:

```python
scores = sl.accuracy(predictions.state, test.state, by=test.subject)
```

This returns a dataset with group values, accuracy, and counts. It does not
average across groups. See [Validation](validation.md) for cross-validation
results and [Evaluation](../reference/evaluation.md) for metric definitions,
averaging, and undefined scores.
