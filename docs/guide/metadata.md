# Metadata

Fields carry information about their values and how to display them.

| Attribute | Meaning |
| --- | --- |
| `dtype` | Value type. |
| `nullable` | Missing values allowed. |
| `dims` | Axis names. |
| `coords` | Axis coordinates. |
| `shape` | Array dimensions. |
| `unit`, `label` | Units and labels. |
| `unit_latex`, `label_latex` | Formatted units and labels. |
| `colour`, `marker` | Series appearance. |
| `colours`, `markers` | Category appearance. |

## Update

```python
import signalearn as sl

data = sl.Dataset({
    "state": sl.Field(["rest", "active"], colours={"rest": "blue", "active": "orange"}),
})
updated = data.update(state={"colours": {"active": "purple"}})
```

`rest` stays blue; `active` becomes purple. `update` returns a new dataset.
Changing a unit declaration does not convert its values.

## Display

Plots use field labels, units, and category colours automatically. Formatted
strings take precedence over plain labels. Mathtext such as
`r"$\mu\mathrm{V}$"` needs no external LaTeX installation.

Use field metadata for meaning and category appearance, and a
[plot style](plotting.md#styles) for fonts, marker sizes, and grids.
See [update](../reference/data.md#update) for merging, clearing, and formatted labels.
