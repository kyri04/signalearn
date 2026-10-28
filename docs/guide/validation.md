# Validation

Cross-validation runs your training procedure on explicit train/test partitions.
Create fresh fitted components inside the procedure so folds do not share state.

This example assumes `data` has `temperature` and `state` fields, with enough
samples in each class for three folds:

```python
import signalearn as sl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def procedure(train, test):
    scaler = sl.fit(train.temperature, StandardScaler())
    train_features = sl.transform(train.temperature, scaler)
    test_features = sl.transform(test.temperature, scaler)
    model = sl.fit(train_features, LogisticRegression(), target=train.state)
    return sl.predict(test_features, model)

splits = sl.folds(
    data, StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    target=data.state,
)
held_out = sl.crossval(splits, procedure, target=data.state)
scores = sl.accuracy(held_out.state, held_out.state_target, by=held_out.fold)
```

## Splits

Native splitters choose memberships. Examples from sklearn include:

| Need | Splitter |
| --- | --- |
| Class balance | `StratifiedKFold` |
| Separate subjects | `GroupKFold` |
| Both | `StratifiedGroupKFold` |
| Time order | `TimeSeriesSplit` |
| Repeated tests | `RepeatedStratifiedKFold` |
| Existing assignments | `PredefinedSplit` |

Pass `target=data.state` for stratification or `group=data.subject` for grouped
splits. Input order, gaps, and restricted training windows are preserved.
Inspect a partition with `train, test = splits[0]`.

## Results

Results include predictions, `fold`, and `source_id`. Supplying `target` adds
aligned labels such as `state_target`; use these for scoring partial or repeated
tests. Repeated predictions receive new event IDs and are not averaged.

Cross-validation freezes each fold's predictions but does not return a final
model. Fit final components separately after choosing settings. Automatic tuning
and nested validation are not implemented. See [folds](../reference/preprocess.md#folds-and-folds)
and [crossval](../reference/evaluation.md#crossval) for the full contracts.
