# Learning

Keep preprocessing and model fitting separate so each step can be inspected
or replaced. Configure native model settings on the component itself.

This example assumes `train` and `test` have numeric `temperature` fields and
class labels in `state`:

```python
import signalearn as sl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = sl.fit(train.temperature, StandardScaler())
train_features = sl.transform(train.temperature, scaler)
test_features = sl.transform(test.temperature, scaler)
model = sl.fit(train_features, LogisticRegression(), target=train.state)
predictions = sl.predict(test_features, model)
```

Fit learned steps on training data only. Apply the fitted components to test
data without refitting. `fit` returns a fitted copy and leaves its input model
unchanged; `transform` and `predict` return datasets with matching sample IDs.

## Predictions

Labels use the target field's name, so this example returns `predictions.state`.
Supported classifiers can also return probabilities or decision scores:

```python
probabilities = sl.predict(test_features, model, output="probabilities")
scores = sl.predict(test_features, model, output="scores")
```

Class coordinates identify each value. Decision scores are not probabilities.
Use label predictions for the existing classification metrics.

## Support

The current sklearn integration accepts scalar features and one target.
Array feature extraction must be explicit. Custom transformations follow the
[component contract](../extensions/components.md); native PyTorch and TensorFlow
learning integrations are not implemented.

See [Learning](../reference/learning.md) for parameters, naming, units, and
execution, and [Predictions](../reference/predictions.md) for output meaning.
