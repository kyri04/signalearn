# Agents

Give your agent access to these docs through a documentation MCP tool or a link
in your project's `AGENTS.md`. SignaLearn does not provide its own MCP server.
Use the reference to check exact APIs and constraints.

## Instructions

Paste this into your project's `AGENTS.md`, replacing the documentation location:

```markdown
## SignaLearn

Consult <documentation URL or local docs path> before writing or modifying
SignaLearn workflows. Use the documentation MCP tool when available.

Keep scientific choices explicit. Use public SignaLearn operations for routine
data handling and keep meaningful steps separately configurable and reusable.
Verify supported inputs, outputs, and constraints in the reference.
```

## Workflow

Establish what each sample represents, its fields and units, existing partitions,
and the independent unit of evaluation. Resolve ambiguity with the source or
user. Preserve original downloads and record preparation decisions.

Keep preprocessing and models separate. Fit learned steps inside training
partitions, and choose grouped or temporal splits when appropriate. Make native
settings, randomness, and assumptions visible.

Do not silently infer targets, convert units, impute values, drop observations,
flatten arrays, or replace source partitions to make code run. Check IDs, shapes,
class order, metadata, and train/test separation. Successful execution alone
does not establish scientific validity.

Use [components](components.md) for custom transformations and
[adapters](adapters.md) for native integrations. Keep reusable project code in
`signalearn_<workspace>` and helpers in its `_internal/` folder. Report unsupported
capabilities and unresolved assumptions.
