# Learning
Keep preprocessing and model fitting separate so each step can be inspected
or replaced. Configure native model settings on the component itself.

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

## Models
Pass sklearn estimators directly. For PyTorch or Keras, configure the native
network and training choices, then wrap it in `sl.Model`. For a compiled Keras
network taking 28-by-28 float32 images and returning ten logits:

```python
component = sl.Model(
    network,
    inputs={"image": {"dims": ("height", "width"), "dtype": "float32"}},
    task="classification",
    classes=list(range(10)),
    output="logits",
    epochs=5,
    batch_size=64,
)
model = sl.fit(train_images.image, component, target=train.digit)
predictions = sl.predict(test_images.image, model)

sl.accuracy(predictions.digit, test.digit)
```

Preprocessing remains a separate `sl.transform` step. Network architecture,
loss, and optimiser stay editable in the native library. See
[Models](../reference/models.md) for complete PyTorch and Keras configurations.

## Support
Sklearn accepts scalar and fixed-length vector features. For example:

```python
from sklearn.decomposition import PCA

pca = sl.fit(train.eeg, PCA(n_components=2))
test_pca = sl.transform(test.eeg, pca)
sl.plot_scatter(test_pca.pca0, test_pca.pca1, by=test.stage)
```

Known scalers preserve vector Fields; PCA produces new component Fields.
Reshape multidimensional inputs explicitly. `sl.Model`
accepts fixed-shape numeric arrays. Both learning integrations use one scalar
target. Custom transformations follow the [component contract](../extensions/components.md).

See [Learning](../reference/learning.md) for parameters, naming, units, and
execution, and [Predictions](../reference/predictions.md) for output meaning.
