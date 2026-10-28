# Adapters
Adapters connect native components to SignaLearn operations. Pass a supported
component to `fit`, `transform`, or `predict`; SignaLearn selects its adapter
internally. Configure native options on the component.

## Support
| Component | Operations |
| --- | --- |
| sklearn-compatible estimator | Fit, transform, predict, save/load. |
| Declared custom transformation | Transform. |

An operation is available only when the component implements its supported
native method.

Sklearn learning currently accepts scalar features and a single target. Native
PyTorch and TensorFlow integrations are not implemented. A placeholder file or
custom wrapper does not provide an integration by itself.

## Responsibilities
Adapters handle native input/output conversion, state, metadata, and supported
execution modes. Native libraries own their algorithms, losses, optimisers,
and learned state. Shared helpers handle alignment and storage; public operations
retain the main input-to-result sequence.

Preserve IDs, dimensions, coordinates, types, units, and class meaning. Equal
shapes do not establish correspondence. Declare whether execution acts on one
sample, a complete batch, or incremental state. Lazy loading alone does not
provide incremental training.

Probabilities, logits, margins, and pairwise scores need distinct declarations.
Preserve class order, score direction, and native label decisions. Reject
unsupported structures instead of reshaping or normalising them implicitly.

Persistence adapters specify the native format, dependencies, importability,
and restoration rules. Common manifests preserve scientific metadata separately.
Do not fall back to arbitrary pickling. Device conversion belongs to native
deployment tools.

Verify every supported operation against native results and document its limits.
See [Learning](../reference/learning.md) and [Predictions](../reference/predictions.md)
for the current contracts.
