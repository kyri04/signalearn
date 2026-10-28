# Selection
Use a field to choose matching samples. The result is a dataset containing
those samples and their fields, with IDs and metadata retained.

```python
selected = sl.filter(data.subject, "S01")
selected = sl.between(data.temperature, lower=36, upper=38)
```

Selection uses the field's current dataset. Filtering a subset never restores
samples excluded earlier. Category values must match the stored values exactly.

## More operations
| Task | Call |
| --- | --- |
| Several categories | `sl.filter(data.subject, ["S01", "S02"])` |
| Missing values | `sl.filter(data.temperature, None)` |
| Custom condition | `sl.where(data.temperature, lambda x: x is not None and x > 37)` |
| Sort | `sl.sort(data.temperature)` |
| Select IDs | `sl.take(data.id, data.ids[:2])` |
| Random sample | `sl.sample(data, n=2, seed=42)` |

## Partitions
Use an existing partition field when one is supplied:

```python
train = sl.filter(data.partition, "train")
test = sl.filter(data.partition, "test")
```

To create new partitions:

```python
train, test = sl.split(data, seed=42)
```

Pass `group=data.subject` to keep subjects together. Use `shuffle=False`
without a seed for sequential splitting. For stratification or more complex
splits, see [Validation](validation.md).

## Combine
```python
combined = sl.concat([train, test])
```

Concatenation preserves the supplied order and rejects duplicate IDs or
incompatible metadata. It does not restore an earlier random order.
See [Preprocessing](../reference/preprocess.md) for exact contracts.
