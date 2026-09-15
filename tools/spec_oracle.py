#!/usr/bin/env python3
"""Independent, exact i64 V0-design semantic oracle; developer tooling only."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
EDITION = "v0-design"
ERROR_IDS = {
    "overflow": "TH-SPEC-I64-OVERFLOW",
    "bounds": "TH-SPEC-BOUNDS",
    "shape": "TH-SPEC-SHAPE",
    "broadcast": "TH-SPEC-BROADCAST",
    "slice": "TH-SPEC-SLICE",
    "axis": "TH-SPEC-AXIS",
}
OPERATIONS = {"construct", "add", "index", "slice", "matmul", "sum"}


class SpecFailure(Exception):
    def __init__(self, error_id):
        self.error_id = error_id
        super().__init__(error_id)


class FixtureFailure(Exception):
    pass


def integer(value, label):
    if type(value) is not int:
        raise FixtureFailure(f"{label} must be an integer")
    return value


def checked(value):
    if value < I64_MIN or value > I64_MAX:
        raise SpecFailure(ERROR_IDS["overflow"])
    return value


def product(shape):
    result = 1
    for extent in shape:
        result *= extent
    return result


def coordinates(shape):
    if not shape:
        yield ()
    elif product(shape):
        for linear in range(product(shape)):
            remaining = linear
            coordinates_reversed = []
            for extent in reversed(shape):
                coordinates_reversed.append(remaining % extent)
                remaining //= extent
            yield tuple(reversed(coordinates_reversed))


@dataclass(frozen=True)
class ScalarValue:
    value: int

    def canonical(self):
        return {"status": "ok", "kind": "scalar", "dtype": "i64", "value": self.value}


@dataclass(frozen=True)
class TensorValue:
    shape: tuple
    values: tuple

    @property
    def rank(self):
        return len(self.shape)

    def offset(self, indices):
        if len(indices) != self.rank:
            raise SpecFailure(ERROR_IDS["shape"])
        offset = 0
        for index, extent in zip(indices, self.shape):
            if index < 0 or index >= extent:
                raise SpecFailure(ERROR_IDS["bounds"])
            offset = offset * extent + index
        return offset

    def at(self, indices):
        return self.values[self.offset(indices)]

    def canonical(self):
        return {"status": "ok", "kind": "tensor", "dtype": "i64",
                "shape": list(self.shape), "values": list(self.values)}


def value(raw):
    if type(raw) is not dict or raw.get("dtype") != "i64":
        raise FixtureFailure("input must be an i64 value object")
    if raw.get("kind") == "scalar" and set(raw) == {"kind", "dtype", "value"}:
        return ScalarValue(checked(integer(raw["value"], "scalar value")))
    if raw.get("kind") == "tensor" and set(raw) == {"kind", "dtype", "shape", "values"}:
        shape, values = raw["shape"], raw["values"]
        if type(shape) is not list or len(shape) not in (1, 2) or type(values) is not list:
            raise FixtureFailure("tensor requires rank-1/2 shape and flat values")
        extents = tuple(integer(x, "extent") for x in shape)
        if any(x < 0 or x > I64_MAX for x in extents):
            raise SpecFailure(ERROR_IDS["shape"])
        if len(values) != product(extents):
            raise SpecFailure(ERROR_IDS["shape"])
        return TensorValue(extents, tuple(checked(integer(x, "tensor element")) for x in values))
    raise FixtureFailure("unknown or malformed value object")


def broadcast_shape(left, right):
    result = []
    for i in range(1, max(len(left), len(right)) + 1):
        a = left[-i] if i <= len(left) else 1
        b = right[-i] if i <= len(right) else 1
        if a != b and a != 1 and b != 1:
            raise SpecFailure(ERROR_IDS["broadcast"])
        result.append(b if a == 1 else a)
    return tuple(reversed(result))


def broadcast_at(operand, result_index):
    if isinstance(operand, ScalarValue):
        return operand.value
    suffix = result_index[-operand.rank:]
    return operand.at(tuple(0 if extent == 1 else index
                            for index, extent in zip(suffix, operand.shape)))


def add(left, right):
    if isinstance(left, ScalarValue) and isinstance(right, ScalarValue):
        return ScalarValue(checked(left.value + right.value))
    shape = broadcast_shape(left.shape if isinstance(left, TensorValue) else (),
                            right.shape if isinstance(right, TensorValue) else ())
    return TensorValue(shape, tuple(checked(broadcast_at(left, index) + broadcast_at(right, index))
                                    for index in coordinates(shape)))


def slice_tensor(tensor, selectors):
    if type(selectors) is not list or len(selectors) != tensor.rank:
        raise SpecFailure(ERROR_IDS["shape"])
    selected = []
    output_shape = []
    for selector, extent in zip(selectors, tensor.shape):
        if type(selector) is int:
            if selector < 0 or selector >= extent:
                raise SpecFailure(ERROR_IDS["bounds"])
            selected.append((selector,))
        elif type(selector) is dict and set(selector) == {"start", "end", "step"}:
            start, end, step = (integer(selector[k], k) for k in ("start", "end", "step"))
            if not (0 <= start <= end <= extent and step > 0):
                raise SpecFailure(ERROR_IDS["slice"])
            positions = tuple(range(start, end, step))
            selected.append(positions)
            output_shape.append(len(positions))
        else:
            raise FixtureFailure("selector must be an integer or explicit slice")
    if not output_shape:
        return ScalarValue(tensor.at(tuple(axis[0] for axis in selected)))
    result = []
    for position in coordinates(tuple(output_shape)):
        source = []
        next_axis = iter(position)
        for selector, positions in zip(selectors, selected):
            source.append(positions[next(next_axis)] if type(selector) is dict else positions[0])
        result.append(tensor.at(tuple(source)))
    return TensorValue(tuple(output_shape), tuple(result))


def matmul(left, right):
    if not isinstance(left, TensorValue) or not isinstance(right, TensorValue) or left.rank != 2 or right.rank != 2:
        raise SpecFailure(ERROR_IDS["shape"])
    m, k = left.shape
    other_k, n = right.shape
    if k != other_k:
        raise SpecFailure(ERROR_IDS["shape"])
    result = []
    for row in range(m):
        for column in range(n):
            total = 0
            for inner in range(k):
                term = checked(left.at((row, inner)) * right.at((inner, column)))
                total = checked(total + term)
            result.append(total)
    return TensorValue((m, n), tuple(result))


def sum_axis(tensor, axis):
    if not isinstance(tensor, TensorValue):
        raise SpecFailure(ERROR_IDS["shape"])
    axis = integer(axis, "axis")
    if axis < 0 or axis >= tensor.rank:
        raise SpecFailure(ERROR_IDS["axis"])
    result_shape = tensor.shape[:axis] + tensor.shape[axis + 1:]
    result = []
    for output_index in coordinates(result_shape):
        total = 0
        for index in range(tensor.shape[axis]):
            source_index = output_index[:axis] + (index,) + output_index[axis:]
            total = checked(total + tensor.at(source_index))
        result.append(total)
    if not result_shape:
        return ScalarValue(result[0])
    return TensorValue(result_shape, tuple(result))


def execute(case):
    inputs = case["inputs"]
    operation = case["operation"]
    try:
        if operation == "construct":
            result = value(inputs["value"])
        elif operation == "add":
            result = add(value(inputs["left"]), value(inputs["right"]))
        elif operation == "index":
            tensor = value(inputs["tensor"])
            indices = inputs["indices"]
            if not isinstance(tensor, TensorValue) or type(indices) is not list:
                raise FixtureFailure("index requires tensor and indices")
            result = ScalarValue(tensor.at(tuple(integer(x, "index") for x in indices)))
        elif operation == "slice":
            tensor = value(inputs["tensor"])
            if not isinstance(tensor, TensorValue):
                raise FixtureFailure("slice requires tensor")
            result = slice_tensor(tensor, inputs["selectors"])
        elif operation == "matmul":
            result = matmul(value(inputs["left"]), value(inputs["right"]))
        elif operation == "sum":
            result = sum_axis(value(inputs["tensor"]), inputs["axis"])
        else:
            raise FixtureFailure(f"unknown operation: {operation}")
        return result.canonical()
    except SpecFailure as failure:
        return {"status": "error", "error_id": failure.error_id}


def validate_fixture(document):
    if type(document) is not dict or set(document) != {"schema", "cases"} or document["schema"] != "thiran-spec-cases-1":
        raise FixtureFailure("invalid fixture envelope")
    cases = document["cases"]
    if type(cases) is not list or not cases:
        raise FixtureFailure("zero executable cases")
    seen = set()
    for case in cases:
        if type(case) is not dict or set(case) != {"id", "edition", "operation", "inputs", "expected", "rationale"}:
            raise FixtureFailure("case missing or containing unexpected fields")
        case_id = case["id"]
        if type(case_id) is not str or not case_id or case_id in seen:
            raise FixtureFailure(f"duplicate or invalid case ID: {case_id}")
        seen.add(case_id)
        if case["edition"] != EDITION or case["operation"] not in OPERATIONS or type(case["inputs"]) is not dict:
            raise FixtureFailure(f"{case_id}: unknown edition/operation or malformed inputs")
        input_fields = {
            "construct": {"value"}, "add": {"left", "right"},
            "index": {"tensor", "indices"}, "slice": {"tensor", "selectors"},
            "matmul": {"left", "right"}, "sum": {"tensor", "axis"},
        }
        if set(case["inputs"]) != input_fields[case["operation"]]:
            raise FixtureFailure(f"{case_id}: invalid operation inputs")
        if type(case["rationale"]) is not str or not case["rationale"]:
            raise FixtureFailure(f"{case_id}: missing rationale")
        expected = case["expected"]
        if type(expected) is not dict:
            raise FixtureFailure(f"{case_id}: expected must be an object")
        if expected.get("status") == "error":
            if set(expected) != {"status", "error_id"} or expected["error_id"] not in ERROR_IDS.values():
                raise FixtureFailure(f"{case_id}: invalid expected error")
        elif expected.get("status") == "ok":
            if expected.get("kind") == "scalar":
                if set(expected) != {"status", "kind", "dtype", "value"} or expected["dtype"] != "i64":
                    raise FixtureFailure(f"{case_id}: invalid expected scalar")
                checked(integer(expected["value"], "expected scalar"))
            elif expected.get("kind") == "tensor":
                if set(expected) != {"status", "kind", "dtype", "shape", "values"} or expected["dtype"] != "i64":
                    raise FixtureFailure(f"{case_id}: invalid expected tensor")
                value({k: expected[k] for k in ("kind", "dtype", "shape", "values")})
            else:
                raise FixtureFailure(f"{case_id}: unknown expected kind")
        else:
            raise FixtureFailure(f"{case_id}: invalid expected status")
    return cases


def validate(path, chosen_id=None):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = validate_fixture(document)
    selected = [case for case in cases if chosen_id is None or case["id"] == chosen_id]
    if not selected:
        raise FixtureFailure(f"no case executed: {chosen_id}")
    failed = 0
    for case in selected:
        actual = execute(case)
        if actual != case["expected"]:
            failed += 1
            print(f"FAIL {case['id']}: expected {json.dumps(case['expected'], sort_keys=True)}; actual {json.dumps(actual, sort_keys=True)}")
    if failed:
        return 1
    print(f"PASS {len(selected)} semantic case(s)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate frozen exact-i64 Thiran V0-design semantic cases.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validator = subparsers.add_parser("validate", help="validate and execute all cases")
    validator.add_argument("fixture", help="JSON case fixture")
    single = subparsers.add_parser("case", help="execute one case from a fixture")
    single.add_argument("case_id")
    single.add_argument("fixture", nargs="?", default="tests/spec/v0/cases.json")
    args = parser.parse_args(argv)
    try:
        return validate(args.fixture, args.case_id if args.command == "case" else None)
    except (FixtureFailure, SpecFailure, OSError, json.JSONDecodeError, KeyError) as failure:
        print(f"fixture error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
