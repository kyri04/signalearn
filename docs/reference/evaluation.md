# Evaluation
Defined in `signalearn.evaluation`. For workflows, see
[metrics](../guide/evaluation.md) and [cross-validation](../guide/validation.md).

## Inputs
Metrics take a prediction Field, or Dataset with exactly one non-ID Field,
then a target Field. Require the same non-empty ID set, aligning target rows
automatically. Inputs are scalar, complete, and have exactly matching units.
String, boolean, integer, and float labels are supported; numeric labels compare
exactly without converting booleans to integers or rounding large integers.
Regression requires numeric Fields. Arrays, multiple outputs, probability/score
Fields, missing values, or incompatible dtypes raise errors. No observations
are dropped. Required inputs are computed together, sharing upstream work.

## accuracy
```python
sl.accuracy(predictions, target, *, by=None)
```

Return the exact-match fraction as a Python float. There is no tolerance,
averaging mode, thresholding, or probability conversion. `by` changes the
return to the grouped Dataset described below.

## Classification
```python
sl.precision(predictions, target, *, average, positive=None, by=None)
sl.recall(predictions, target, *, average, positive=None, by=None)
sl.sensitivity(predictions, target, *, average, positive=None, by=None)
sl.specificity(predictions, target, *, average, positive=None, by=None)
sl.f1(predictions, target, *, average, positive=None, by=None)
```

Each class is evaluated against all others. Precision is TP/(TP+FP), recall
TP/(TP+FN), specificity TN/(TN+FP), and F1 2TP/(2TP+FP+FN). Sensitivity is the
exact recall alias, including signature and `recall` result name. Binary
specificity equals recall for the opposite class; multiclass specificity
remains one class versus all others.

### Averaging
| Required `average` | Return |
| --- | --- |
| `"macro"` | Unweighted class mean. |
| `"micro"` | Ratio after pooling relevant counts. |
| `"weighted"` | Class mean weighted by true-target support. |
| `"binary"` | Score for explicit `positive`. |
| `None` | Dataset with `category`, metric, and `support`. |

Use the sorted union of observed true/predicted classes. Metadata-only categories
are not added. Predicted-only classes have zero support. Binary mode permits at
most two classes including the supplied compatible finite positive label; that
label may be absent from observations. Passing positive in another mode errors.
There is no inferred average or positive class.

Per-class summary IDs are new. Category metadata comes from the target,
including labels, units, formatted metadata, colours, and markers. Scores are
dimensionless. Categories must share a lossless scalar dtype; a category Field
that would round a large integer is rejected, even though aggregate scores and
confusion plots can still distinguish the original labels.

### Undefined values
A zero denominator warns. Aggregate undefined scores are NaN; per-class scores
are `None` in nullable Fields. Macro propagates undefined values. Weighted
averages ignore zero-support classes, but propagate undefined values with
positive support. Micro is undefined only if its pooled denominator is zero.
F1 is computed directly from counts: with some FP/FN and no TP, it is zero,
even when precision or recall is undefined.

## Regression
```python
sl.mae(predictions, target, *, by=None)
sl.rmse(predictions, target, *, by=None)
sl.r2(predictions, target, *, by=None)
```

Return Python floats without by. MAE is mean absolute error; RMSE is root mean
squared error. Both use the matching value unit. R2 is one minus residual sum
of squares divided by target-deviation sum of squares; it is dimensionless and
can be negative. Integer residuals are calculated before conversion to retain
small differences between large integers; scaling avoids unnecessary overflow.
Unrepresentable float64 differences raise errors.

R2 warns and returns NaN with fewer than two observations. A constant target
warns and returns NaN for perfect predictions or negative infinity otherwise,
following the mathematical behaviour of sklearn `force_finite=False`.

## Groups
Every metric accepts optional `by`, a scalar complete Field with the exact
prediction IDs. Targets and groups are aligned; missing values or ID subsets/
supersets are errors. Shared inputs are computed together once before groups
are scored independently. Class unions and undefined-score rules apply within
each group, rather than importing other groups' categories.

Aggregate grouped results contain the grouping Field, metric, and integer
`count`, with new IDs. Groups are sorted by scalar value. Per-class grouping
(`average=None`) yields group/class rows, `category`, metric, and `support`,
without an additional count Field. Dtypes and grouping/target metadata survive.
The grouping name defaults to `group`; collisions with `id`, `category`,
`support`, `count`, or the metric append `_group`. MAE/RMSE summaries retain
target unit and formatted unit; other metric Fields are dimensionless.

Undefined group scores warn and become `None`, retaining the group/count,
including non-finite R2 values. Invalid inputs still fail the whole operation.
No average across groups is inferred. A pooled metric and equal-weight group
mean are distinct analysis choices.

## crossval
```python
sl.crossval(splits, procedure, *, target=None)
```

`splits` is a [Folds plan](preprocess.md#folds-and-folds). Invoke
`procedure(train, test)` sequentially for each pair. It must return a Dataset
with at least one prediction Field and exactly the test IDs; order is aligned
automatically. Extra/missing/new IDs, incompatible result schemas, or reserved
`fold`/`source_id` names, including hidden coordinate names, are errors.

Fit learned steps on the training partition, starting fresh state or the same
pretrained checkpoint in every fold. Callable objects are deep-copied from their
initial state per split. Python functions retain globals/closures; the caller
must avoid shared mutable fitted state. Non-copyable training state can instead
be created inside a function.

Every fold's returned Fields and coordinate dependencies are eagerly computed
and copied before the next fold, so later result access does not rerun models.
Results must fit in memory. Fitted models are not collected. No automatic
tuning, nesting, or final model retraining is performed. Failures during a fold
include its split context.

### Results
The result contains returned predictions plus zero-based `fold` and `source_id`
from the plan input. With unrepeated test coverage, preserve input IDs and input
order, including partial coverage. If any sample is tested repeatedly, all
prediction rows get new event IDs and follow split then native test order.
No averaging or deduplication occurs. The native splitter determines repeat
structure; folds are indexed in its supplied order.

Optional `target` is a complete scalar exact-ID input Field. Its held-out values
become `<target>_target`, with an automatic suffix on collision. This reference
preserves metadata and is repeated with prediction events. It does not select
the procedure's training target. Use reference Fields to evaluate partial or
repeated coverage without manually matching the original target.

For windowed input, result source_id identifies the input window. Recover the
original recording/subject relationship from that window Dataset using IDs.
Return prediction Fields rather than copying reserved window provenance into
the procedure result.

Probability/score metadata is preserved. Equivalent per-class columns align by
label to first-fold order. Different vocabularies, missing classes, incompatible
meaning, or reversed scalar score direction are rejected. Pooled metrics count
every prediction event, which may weight source samples unevenly; use `by=fold`
for independent fold summaries.
