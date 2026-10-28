# Learning
`fit` and `predict` live in `signalearn.learning`; `transform` lives in
`signalearn.preprocess`. All are exported at the package root.

## Inputs
`x` is a Field, a sequence of Fields, or a Dataset. A Dataset supplies every
field except `id`; select features explicitly to avoid including targets or
unrelated metadata. Require non-empty input and distinct names other than `id`.
Unnamed Fields use `x`, or `x0`, `x1`, etc. All IDs must match exactly; rows align
to the first selected Field.

The native sklearn integration requires scalar inputs. One Field becomes one
matrix column. Numeric missing values become NaN; categorical missing values
remain `None`. Native code either supports them or errors. No observations are
dropped or imputed. Integer identity is preserved during mixed column conversion.

Native options belong on the component. Extra operation arguments forward to
its native method. Field-valued arguments align by ID; raw arrays use native
positional semantics. Ordinary parameters are copied for deferred operations.

## fit
```python
sl.fit(x, model, target=None, **fit_params)
```

Return a fitted native copy. The component implements `get_params` and `fit`,
with fit returning itself. Configuration is cloned, including random seeds;
previous fitted state is not reused and the supplied instance is unchanged.
Use `target` for a scalar, complete, exact-ID supervised target, passed as a
one-dimensional native array. Passing native `y` through `fit_params` is an error.

Compute selected inputs, target, and Field-valued parameters together, then call
native fit once on the complete selected batch. This is eager and not incremental
or out of core. No training-resumption contract is inferred.

Record input names, numeric/boolean/string kinds, and units. Integer and float
features share the numeric kind. Target metadata records its name, dtype, units,
labels, and display preferences, without retaining target observations in that
metadata. The native model itself may retain observations.

## transform
```python
sl.transform(x, model, **transform_params)
```

Return a lazy Dataset of transformed Fields with original IDs. Applying never
calls fit or fit_transform. Native components must support transform and be
fitted when required. Custom scalar/array components use the
[declared contract](../extensions/components.md).

For SignaLearn-fitted native components, input names must match training names;
columns are reordered to fitted order. Logical kind and units must match.
Native feature names are used for externally fitted components when available;
otherwise supply the original positional order and units because SignaLearn
has no scientific training metadata. Fitted feature counts are checked.

The component and ordinary parameters are snapshotted at the call. Native
transformation executes once per joint computation over the complete selection;
separate value accesses may recompute. Native output is real numeric, with a
sample axis and declared column count. A vector means one column. Sample count
and order must be retained. Sparse numeric output is densified internally.
Fields are nullable `float64`: NaNs become `None`, infinity/non-numeric values
and integers losing float64 precision are rejected.

### Outputs
Use `get_feature_names_out` when provided. An identity FunctionTransformer
retains names. Otherwise the component must declare positive `n_features_out_`:
names use the snake-case class plus source for one input/output, or class plus
output index. For example `Square` can generate `square_amplitude`.
Missing schemas fail before executing data; matching column counts do not
establish unchanged scientific meaning. `FunctionTransformer` can explicitly
declare `feature_names_out="one-to-one"` or an appropriate naming callback.

Outputs declared with unchanged source names retain labels, formatted labels,
series colours, and markers. Category mappings are cleared. New features start
without inherited units/display preferences. StandardScaler with scaling clears
units; centring without scaling, RobustScaler without scaling, and identity
FunctionTransformer retain them. Other unknown units remain unspecified.
Changing units or feature meaning requires adapter/component knowledge.

## predict
```python
sl.predict(x, model, *, output=None, **predict_params)
```

Return a lazy single-target Dataset with original IDs. Input checks, ordering,
native argument alignment, and snapshots follow transform. Target values are
not required. The adapter supports native sklearn classifier/regressor tags;
a predict method alone does not establish task meaning. Known multioutput
schemas are rejected before executing values.

| `output` | Native method | Field |
| --- | --- | --- |
| `None` | `predict` | Target-named label or regression value. |
| `"probabilities"` | `predict_proba` | `<target>_probability` |
| `"scores"` | `decision_function` | `<target>_score` |

Default class labels retain string/boolean/integer/float type and valid target
display metadata. Labels must belong to native `classes_`. Regression uses
finite `float64`, retaining target units/series display and clearing category
maps. Native output must be one value per sample, as a vector or single column.
Missing/invalid values, wrong sample counts, or precision loss raise errors.

An unnamed target, target named `id`, or externally fitted component uses base
name `prediction`. External components have no inferred target units or labels.
Default prediction always preserves native decisions, even when they differ
from probability argmax. Alternative outputs require supported native methods;
no calibration, softmax, or model refitting is automatic. Full class/direction
contracts are in [prediction outputs](predictions.md).
