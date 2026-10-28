# Saving

Save datasets, fitted components, and figures explicitly. Destinations must be
new; SignaLearn creates missing parent folders and rejects existing paths.

## Data

For a dataset named `data`:

```python
path = sl.save_data(data, "results/data")
restored = sl.load_data(path)
```

The result contains `samples.csv`, `samples.schema.json`, and `arrays/` when
needed. Share the whole folder. IDs, metadata, coordinates, and prediction
meaning survive loading. Saving computes the selected values.

## Models

For separately fitted `scaler` and `model` components:

```python
sl.save_model(scaler, "models/scaler")
sl.save_model(model, "models/classifier")

scaler = sl.load_model("models/scaler")
model = sl.load_model("models/classifier")
```

Save and restore preprocessing alongside the classifier. Supported sklearn
artifacts contain `model.json` and `model.joblib`. Dependencies must be compatible,
and custom definitions must remain importable.

**Load model artifacts only from trusted sources.** Joblib uses pickle;
manifest checks do not make an untrusted payload safe. These files support
compatible Python environments, not automatic device conversion.

## Figures

For a plotting result named `fig`:

```python
sl.save_plot(fig, "figures/result.pdf")
```

Pass a Figure or the full `(figure, axes)` result. The extension chooses the
format; style settings control export. The figure stays open. Tight cropping
can change output bounds; use `savefig.bbox: standard` for the full canvas.

See [I/O](../reference/io.md) for dataset/model contracts and
[save_plot](../reference/plotting.md#save_plot) for figure export.
