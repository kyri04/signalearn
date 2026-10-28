import ast
import io
from keyword import iskeyword
import sys
import tokenize
from types import MappingProxyType
from typing import NamedTuple

_MATCHER_ID = "signalearn.indexed_fields"


class CompletionView(NamedTuple):
    kind: type
    fields: tuple[str, ...]
    length: int


def object_view(value) -> CompletionView | None:
    from ..data import Dataset, Field, Sample, SampleField

    if type(value) is Dataset:
        return CompletionView(Dataset, tuple(value._fields), len(value._ids))
    if type(value) is Sample:
        return CompletionView(Sample, tuple(value._dataset._fields), 1)
    if type(value) in (Field, SampleField):
        return CompletionView(type(value), (), len(value.ids))
    return None


def type_attributes(kind: type) -> set[str]:
    names = set(dir(kind))
    for base in kind.__mro__:
        names.update(getattr(base, "__annotations__", {}))
    return names


def literal_selector(node):
    if isinstance(node, ast.Slice):
        bounds = [None if bound is None else ast.literal_eval(bound)
                  for bound in (node.lower, node.upper, node.step)]
        if any(bound is not None and type(bound) is not int for bound in bounds):
            raise ValueError("Slice bounds must be literal integers")
        return slice(*bounds)
    value = ast.literal_eval(node)
    if type(value) in (int, str):
        return value
    if type(value) in (list, tuple) and all(type(item) in (int, bool) for item in value):
        return value
    raise ValueError("Unsupported completion selector")


def selection_size(selector, length: int) -> tuple[int, bool]:
    if type(selector) is int:
        if not -length <= selector < length:
            raise IndexError("Sample index out of range")
        return 1, True
    if isinstance(selector, slice):
        return len(range(*selector.indices(length))), False
    if selector and all(type(index) is bool for index in selector):
        if len(selector) != length:
            raise ValueError("Boolean mask length must match the sample count")
        return sum(selector), False
    normalized = set()
    for index in selector:
        if type(index) is not int or not -length <= index < length:
            raise ValueError("Invalid sample index")
        normalized.add(index + length if index < 0 else index)
    if len(normalized) != len(selector):
        raise ValueError("Selection cannot repeat sample IDs")
    return len(selector), False


def resolve_view(node, namespace) -> CompletionView | None:
    from ..data import Dataset, Field, Sample, SampleField

    if isinstance(node, ast.Name):
        return object_view(namespace.get(node.id))
    if isinstance(node, ast.Attribute):
        parent = resolve_view(node.value, namespace)
        if parent is not None and parent.kind in (Dataset, Sample) and node.attr == "fields":
            return CompletionView(MappingProxyType, parent.fields, parent.length)
        if (parent is not None and parent.kind in (Dataset, Sample) and not node.attr.startswith("_")
                and node.attr in parent.fields
                and node.attr not in type_attributes(parent.kind)):
            return CompletionView(Field if parent.kind is Dataset else SampleField, (), parent.length)
        return None
    if not isinstance(node, ast.Subscript):
        return None
    parent = resolve_view(node.value, namespace)
    if parent is None:
        return None
    selector = literal_selector(node.slice)
    if isinstance(selector, str):
        if parent.kind in (Dataset, MappingProxyType) and selector in parent.fields:
            return CompletionView(Field, (), parent.length)
        if parent.kind is Sample and selector in parent.fields:
            return CompletionView(SampleField, (), 1)
        return None
    if parent.kind in (Sample, MappingProxyType):
        return None
    length, scalar = selection_size(selector, parent.length)
    if parent.kind is Dataset:
        return CompletionView(Sample if scalar else Dataset, parent.fields, length)
    if scalar:
        return None
    return CompletionView(Field, (), length)


def attribute_expression(source: str):
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.ERRORTOKEN and not token.string.isspace():
                return None
            if token.type not in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL,
                                  tokenize.INDENT, tokenize.DEDENT):
                tokens.append(token)
    except (tokenize.TokenError, IndentationError):
        pass
    prefix = ""
    if tokens and tokens[-1].type == tokenize.NAME:
        prefix = tokens.pop().string
    if not tokens or tokens[-1].string != ".":
        return None
    dot = tokens.pop()
    lines = source.splitlines()
    line = lines[dot.start[0] - 1]
    for index in range(max(0, len(tokens) - 128), len(tokens)):
        token = tokens[index]
        if token.type != tokenize.NAME or token.start[0] != dot.start[0]:
            continue
        if index and tokens[index - 1].string == ".":
            continue
        expression = line[token.start[1]:dot.start[1]]
        if len(expression) > 2048:
            continue
        try:
            tree = ast.parse(expression, mode="eval").body
        except SyntaxError:
            continue
        if sum(1 for _ in ast.walk(tree)) <= 64:
            return tree, prefix
    return None


def attribute_names(source: str, namespace) -> tuple[list[str], str]:
    expression = attribute_expression(source)
    if expression is None:
        return [], ""
    tree, prefix = expression
    try:
        view = resolve_view(tree, namespace)
    except (ValueError, TypeError, IndexError, KeyError):
        return [], prefix
    if view is None:
        return [], prefix
    names = type_attributes(view.kind)
    if view.kind is not MappingProxyType:
        names |= set(view.fields)
    return sorted(name for name in names
                  if name.isidentifier() and not iskeyword(name)
                  and name.startswith(prefix)
                  and (not name.startswith("_") or prefix.startswith("_"))), prefix


def install_ipython_completion(shell=None) -> bool:
    if shell is None:
        ipython = sys.modules.get("IPython")
        if ipython is None:
            return False
        shell = ipython.get_ipython()
    if shell is None:
        return False
    completer = getattr(shell, "Completer", None)
    if completer is None or not hasattr(completer, "custom_matchers"):
        return False
    if any(getattr(matcher, "matcher_identifier", None) == _MATCHER_ID
           for matcher in completer.custom_matchers):
        return True
    try:
        from IPython.core.completer import SimpleCompletion, completion_matcher
    except ImportError:
        return False

    @completion_matcher(api_version=2, priority=100, identifier=_MATCHER_ID)
    def matcher(context):
        lines = context.full_text.split("\n")[:context.cursor_line]
        source = "\n".join([*lines, context.text_until_cursor])
        namespace = {**shell.user_global_ns, **shell.user_ns}
        names, prefix = attribute_names(source, namespace)
        return {"completions": [SimpleCompletion(name, type="attribute") for name in names],
                "matched_fragment": prefix, "suppress": bool(names)}

    completer.custom_matchers.append(matcher)
    return True
