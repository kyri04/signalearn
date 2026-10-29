import csv
import json
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from dask import compute

from ._internal.data import dataset_dependencies, new_ids
from ._internal.adapters import get_adapter, get_persistence_adapter
from ._internal.array_io import array_storage, source_columns
from ._internal.io import data_paths, field_options, read_csv_columns, read_schema
from ._internal.model_io import (
    check_model_versions, model_manifest, model_payload, read_model_manifest,
    validate_model_metadata, validate_restored_model, write_model_manifest,
)
from ._internal.saving import (
    save_batches, save_destination, save_headers, save_rows, save_tasks, saved_schema,
)
from .data import Dataset, Field

__all__ = ["load_data", "save_data", "save_model", "load_model"]


def save_model(model: Any, path: str | PathLike[str]) -> Path:
    destination = save_destination(path, kind="Model")
    adapter = get_adapter(model, "save_model")
    adapter.validate_persistence(model)
    manifest = model_manifest(model, adapter, adapter.persistence_metadata(model))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        directory = Path(temporary)
        adapter.dump_model(model, directory / adapter.persistence_payload)
        write_model_manifest(directory, manifest)
        save_destination(destination, kind="Model")
        directory.rename(destination)
    return destination


def load_model(path: str | PathLike[str]) -> Any:
    directory, manifest = read_model_manifest(path)
    adapter = get_persistence_adapter(manifest["adapter"])
    payload = model_payload(directory, manifest, adapter)
    check_model_versions(manifest["versions"], adapter)
    metadata = validate_model_metadata(manifest["metadata"], encoded=True)
    model = adapter.load_model(payload)
    validate_restored_model(model, manifest, adapter)
    adapter.restore_metadata(model, metadata)
    return model


def load_data(
    path: str | PathLike[str], *, schema: str | PathLike[str] | None = None,
) -> Dataset:
    path, schema_path = data_paths(path, schema)
    declaration = read_schema(schema_path)
    declarations = declaration["fields"] if declaration["layout"] == "rows" else declaration["sample_fields"]
    columns = read_csv_columns(path, declarations, set(declarations) | source_columns(declaration))
    identity = columns.get("id")
    if identity is None:
        identity = new_ids(len(next(iter(columns.values()))))
    fields = {
        name: Field(values, ids=identity, **field_options(declarations[name]))
        for name, values in columns.items() if name in declarations
    }
    if declaration["layout"] == "arrays":
        headers = {}
        for name, options in declaration["fields"].items():
            storage = array_storage(path, columns, options, identity, name, headers)
            fields[name] = Field(storage, ids=identity, dims=options["dims"],
                                 coords=options.get("coords"), shape=options.get("shape"),
                                 **field_options(options))
    data = Dataset(fields) if "id" in fields else Dataset(fields, ids=identity)
    return data.select(*declaration["visible_fields"]) if "visible_fields" in declaration else data


def save_data(data: Dataset, path: str | PathLike[str]) -> Path:
    if not isinstance(data, Dataset):
        raise TypeError("save_data requires a Dataset")
    destination = save_destination(path)
    fields = dataset_dependencies(data)
    schema = saved_schema(data, fields)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        directory = Path(temporary)
        if schema["layout"] == "arrays":
            (directory / "arrays").mkdir()
        with (directory / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=save_headers(schema))
            writer.writeheader()
            for indices in save_batches(fields, len(data)):
                values = compute(*save_tasks(fields, indices))
                writer.writerows(save_rows(directory, fields, schema, indices, values))
        (directory / "samples.schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        save_destination(destination)
        directory.rename(destination)
    return destination / "samples.csv"
