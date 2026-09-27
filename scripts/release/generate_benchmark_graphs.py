#!/usr/bin/env python3
"""Generate deterministic, dependency-free SVG charts from TH-023 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from html import escape
from typing import Iterable


EXPECTED_SHA256 = "04f7c9d26d4dd749d308328dd69bd5f83f0c4c3786fe380de65a7d4fc0858e31"
WIDTH = 960
HEIGHT = 560
SIZES = ((1024, "small · 1,024"), (1048576, "medium · 1,048,576"),
         (16777216, "large · 16,777,216"))
COLORS = ("#0969da", "#cf222e", "#1a7f37", "#9a6700", "#8250df", "#bf3989")
GRAPH_FILES = (
    "cpu-elementwise-median.svg",
    "gpu-transfer-inclusive-median.svg",
    "fusion-effect.svg",
    "source-vs-artifact.svg",
    "extension-performance.svg",
    "deployment-overhead.svg",
)


@dataclass(frozen=True)
class AuditRow:
    graph: str
    series: str
    scope: str
    raw: str
    rendered: str


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: pathlib.Path) -> list[dict]:
    actual = sha256(path)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"TH-023 SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("checkpoint") != "TH-023" or not isinstance(payload.get("records"), list):
        raise RuntimeError("TH-023 payload has an unexpected schema")
    return payload["records"]


def select(records: list[dict], **fields: object) -> dict:
    matches = [record for record in records
               if all(record.get(key) == value for key, value in fields.items())]
    if len(matches) != 1:
        criteria = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        raise RuntimeError(f"required TH-023 row must be unique ({criteria}); found {len(matches)}")
    record = matches[0]
    median = record.get("median_ns")
    if not isinstance(median, (int, float)) or median <= 0:
        raise RuntimeError(f"required row has invalid median_ns: {criteria}")
    if not isinstance(record.get("samples_ns"), list) or not record["samples_ns"]:
        raise RuntimeError(f"required row has no samples: {criteria}")
    return record


def ms(record: dict) -> float:
    return float(record["median_ns"]) / 1_000_000.0


def display(value: float) -> str:
    return f"{value:.6g}"


def axis_label(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:g}"
    if value >= 0.001:
        return f"{value:.3g}"
    return f"{value:.1e}"


def xml_text(x: float, y: float, value: str, css: str = "fg", anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" '
            f'text-anchor="{anchor}">{escape(value)}</text>')


def header(title: str, description: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc">',
        f'<title id="chart-title">{escape(title)}</title>',
        f'<desc id="chart-desc">{escape(description)}</desc>',
        '<style>',
        ':root{color-scheme:light dark}.bg{fill:#ffffff}.fg{fill:#24292f}.muted{fill:#57606a}'
        '.grid{stroke:#d0d7de;stroke-width:1}.axis{stroke:#57606a;stroke-width:1.2}'
        '.title{font:600 22px system-ui,sans-serif}.subtitle{font:14px system-ui,sans-serif}'
        '.fg,.muted{font:13px system-ui,sans-serif}.legend{font:12.5px system-ui,sans-serif}'
        '.value{font:600 12px ui-monospace,SFMono-Regular,Consolas,monospace}',
        '@media(prefers-color-scheme:dark){.bg{fill:#0d1117}.fg{fill:#e6edf3}.muted{fill:#8b949e}'
        '.grid{stroke:#30363d}.axis{stroke:#8b949e}}',
        '</style>',
        f'<rect class="bg" width="{WIDTH}" height="{HEIGHT}" rx="8"/>',
    ]


def footer(lines: list[str]) -> None:
    lines.append(xml_text(24, 538, f"Source: TH-023-machine.json · SHA-256 {EXPECTED_SHA256}", "muted"))
    lines.append('</svg>')


def log_ticks(values: Iterable[float]) -> list[float]:
    values = list(values)
    low = math.floor(math.log10(min(values)))
    high = math.ceil(math.log10(max(values)))
    return [10.0 ** exponent for exponent in range(low, high + 1)]


def line_chart(path: pathlib.Path, title: str, subtitle: str, series: list[tuple[str, list[dict]]],
               description: str, graph_name: str) -> list[AuditRow]:
    all_values = [ms(record) for _, rows in series for record in rows]
    ticks = log_ticks(all_values)
    low_log, high_log = math.log10(ticks[0]), math.log10(ticks[-1])
    x0, y0, plot_w, plot_h = 90.0, 132.0, 650.0, 340.0
    lines = header(title, description)
    lines.extend((xml_text(24, 35, title, "title"), xml_text(24, 61, subtitle, "subtitle muted")))

    def y(value: float) -> float:
        return y0 + plot_h * (high_log - math.log10(value)) / (high_log - low_log)

    for tick in ticks:
        ty = y(tick)
        lines.append(f'<line x1="{x0}" y1="{ty:.1f}" x2="{x0 + plot_w}" y2="{ty:.1f}" class="grid"/>')
        lines.append(xml_text(x0 - 10, ty + 4, axis_label(tick), "muted", "end"))
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" class="axis"/>')
    axis_y = y0 + plot_h / 2
    lines.append(
        f'<text x="24.0" y="{axis_y:.1f}" class="muted" text-anchor="middle" '
        f'transform="rotate(-90 24.0 {axis_y:.1f})">median milliseconds · logarithmic</text>')

    xs = [x0 + index * plot_w / (len(SIZES) - 1) for index in range(len(SIZES))]
    for xpos, (_, size_label) in zip(xs, SIZES):
        first, second = size_label.split(" · ")
        lines.append(xml_text(xpos, y0 + plot_h + 24, first, "fg", "middle"))
        lines.append(xml_text(xpos, y0 + plot_h + 42, second, "muted", "middle"))

    audit: list[AuditRow] = []
    for series_index, (label, rows) in enumerate(series):
        color = COLORS[series_index]
        points = " ".join(f"{xpos:.1f},{y(ms(record)):.1f}" for xpos, record in zip(xs, rows))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        for xpos, record, (_, size_label) in zip(xs, rows, SIZES):
            value = ms(record)
            rendered = display(value)
            title_text = f"{label} · {size_label}: {rendered} ms"
            lines.append(
                f'<circle cx="{xpos:.1f}" cy="{y(value):.1f}" r="5" fill="{color}" '
                f'data-benchmark="{escape(record["benchmark_name"])}" '
                f'data-implementation="{escape(record["implementation"])}" '
                f'data-backend="{escape(record["backend"])}" data-size="{record["size"]}" '
                f'data-timing-scope="{escape(record["timing_scope"])}" '
                f'data-median-ns="{record["median_ns"]}" data-rendered-ms="{rendered}">'
                f'<title>{escape(title_text)}</title></circle>')
            audit.append(AuditRow(graph_name, label, size_label,
                                  f'{record["median_ns"]} ns', f"{rendered} ms"))
        legend_y = 136 + series_index * 31
        lines.append(f'<line x1="755" y1="{legend_y}" x2="779" y2="{legend_y}" '
                     f'stroke="{color}" stroke-width="2.5"/>')
        lines.append(f'<circle cx="767" cy="{legend_y}" r="4" fill="{color}"/>')
        lines.append(xml_text(790, legend_y + 4, label, "legend fg"))
    footer(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return audit


def horizontal_chart(path: pathlib.Path, title: str, subtitle: str,
                     items: list[tuple[str, float, dict | None, str]],
                     description: str, graph_name: str, unit: str,
                     logarithmic: bool, zero_based: bool = False) -> list[AuditRow]:
    values = [value for _, value, _, _ in items]
    x0, x1 = 295.0, 900.0
    y0, row_h = 142.0, 68.0
    lines = header(title, description)
    lines.extend((xml_text(24, 35, title, "title"), xml_text(24, 61, subtitle, "subtitle muted")))

    if logarithmic:
        ticks = log_ticks(values)
        low, high = math.log10(ticks[0]), math.log10(ticks[-1])
        position = lambda value: x0 + (x1 - x0) * (math.log10(value) - low) / (high - low)
    else:
        maximum = max(values)
        high = math.ceil(maximum * 1.12 * 10.0) / 10.0 if maximum < 10 else math.ceil(maximum * 1.12 / 50.0) * 50.0
        ticks = [high * index / 5.0 for index in range(6)]
        position = lambda value: x0 + (x1 - x0) * value / high

    chart_bottom = y0 + row_h * len(items) - 20
    for tick in ticks:
        xpos = position(tick)
        lines.append(f'<line x1="{xpos:.1f}" y1="{y0 - 28}" x2="{xpos:.1f}" y2="{chart_bottom}" class="grid"/>')
        lines.append(xml_text(xpos, chart_bottom + 24, axis_label(tick), "muted", "middle"))
    axis_title = f"median {unit}" + (" · logarithmic" if logarithmic else " · zero-based")
    lines.append(xml_text((x0 + x1) / 2, chart_bottom + 48, axis_title, "muted", "middle"))

    audit: list[AuditRow] = []
    for index, (label, value, record, scope) in enumerate(items):
        ypos = y0 + index * row_h
        color = COLORS[index % len(COLORS)]
        xpos = position(value)
        lines.append(xml_text(x0 - 18, ypos + 5, label, "fg", "end"))
        if zero_based:
            lines.append(f'<line x1="{x0}" y1="{ypos}" x2="{xpos:.1f}" y2="{ypos}" '
                         f'stroke="{color}" stroke-width="8" stroke-linecap="round" opacity="0.65"/>')
        else:
            lines.append(f'<line x1="{x0}" y1="{ypos}" x2="{xpos:.1f}" y2="{ypos}" '
                         f'stroke="{color}" stroke-width="2" opacity="0.6"/>')
        rendered = display(value)
        if record is not None:
            metadata = (f'data-benchmark="{escape(record["benchmark_name"])}" '
                        f'data-implementation="{escape(record["implementation"])}" '
                        f'data-backend="{escape(record["backend"])}" data-size="{record["size"]}" '
                        f'data-timing-scope="{escape(record["timing_scope"])}" '
                        f'data-median-ns="{record["median_ns"]}" data-rendered-ms="{rendered}"')
            raw = f'{record["median_ns"]} ns'
            rendered_value = f"{rendered} {unit}"
        else:
            metadata = (f'data-derived-label="{escape(label)}" '
                        f'data-derived-value="{value:.12f}" data-rendered-value="{rendered}"')
            raw = f"{value:.12f}x"
            rendered_value = f"{rendered}x"
        lines.append(f'<circle cx="{xpos:.1f}" cy="{ypos}" r="7" fill="{color}" {metadata}>'
                     f'<title>{escape(label)} · {escape(scope)}: {escape(rendered_value)}</title></circle>')
        value_x = min(xpos + 12, 890)
        anchor = "start" if xpos < 845 else "end"
        lines.append(xml_text(value_x if anchor == "start" else xpos - 12, ypos + 4,
                              rendered_value, "value fg", anchor))
        audit.append(AuditRow(graph_name, label, scope, raw, rendered_value))
    footer(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return audit


def generate(records: list[dict], output: pathlib.Path) -> list[AuditRow]:
    output.mkdir(parents=True, exist_ok=True)
    audit: list[AuditRow] = []
    cpu_series = (
        ("Thiran fused", "thiran_fused"),
        ("Thiran unfused", "thiran_unfused"),
        ("optimized C++ fused", "cpp_fused"),
        ("optimized C++ materialized", "cpp_materialized"),
        ("NumPy eager", "numpy_eager"),
        ("PyTorch eager · 1 thread", "pytorch_eager_single_thread"),
    )
    cpu_rows = [(label, [select(records, benchmark_name="elementwise_chain", implementation=impl,
                                backend="cpu", size=size, timing_scope="warm_in_process_host_wall")
                         for size, _ in SIZES]) for label, impl in cpu_series]
    audit += line_chart(output / GRAPH_FILES[0], "CPU elementwise medians",
                        "Qualified f32 chain · warm in-process host wall time · log scale",
                        cpu_rows, "Median CPU times for six comparable implementations over three sizes.",
                        GRAPH_FILES[0])

    gpu_series = (
        ("Thiran fused · transfer-inclusive", "thiran_fused"),
        ("Thiran unfused · transfer-inclusive", "thiran_unfused"),
        ("PyTorch · transfer-inclusive", "pytorch_eager_transfer_inclusive"),
    )
    gpu_rows = [(label, [select(records, benchmark_name="elementwise_chain", implementation=impl,
                                backend="gpu", size=size,
                                timing_scope="transfer_inclusive_synchronized_host_wall")
                         for size, _ in SIZES]) for label, impl in gpu_series]
    audit += line_chart(output / GRAPH_FILES[1], "GPU transfer-inclusive medians",
                        "Synchronized host wall time · H2D + execution + D2H · log scale",
                        gpu_rows, "Transfer-inclusive GPU medians; device-resident PyTorch timing is excluded.",
                        GRAPH_FILES[1])

    fusion_items = []
    for size, label in SIZES:
        fused = select(records, benchmark_name="elementwise_chain", implementation="thiran_fused",
                       backend="cpu", size=size, timing_scope="warm_in_process_host_wall")
        unfused = select(records, benchmark_name="elementwise_chain", implementation="thiran_unfused",
                         backend="cpu", size=size, timing_scope="warm_in_process_host_wall")
        ratio = float(unfused["median_ns"]) / float(fused["median_ns"])
        fusion_items.append((label, ratio, None, "unfused median ÷ fused median"))
    audit += horizontal_chart(output / GRAPH_FILES[2], "TH-015 CPU fusion effect",
                              "Qualified elementwise chain · median speedup · GPU timing remained inconclusive",
                              fusion_items, "CPU median speedup derived as unfused time divided by fused time.",
                              GRAPH_FILES[2], "speedup", False, True)

    path_items = []
    for label, backend, benchmark, implementation in (
        ("CPU source-run", "cpu", "source_run", "thiran_cli_run"),
        ("CPU artifact-run", "cpu", "artifact_run", "thiran_cli_artifact_run"),
        ("GPU source-run", "gpu", "source_run", "thiran_cli_run"),
        ("GPU artifact-run", "gpu", "artifact_run", "thiran_cli_artifact_run"),
    ):
        record = select(records, benchmark_name=benchmark, implementation=implementation,
                        backend=backend, size=32)
        path_items.append((label, ms(record), record, "fresh process"))
    audit += horizontal_chart(output / GRAPH_FILES[3], "Source-run vs artifact-run",
                              "Fresh-process startup/execution-path cost · not kernel-speed evidence",
                              path_items, "Fresh-process source and artifact runtime medians for CPU and GPU.",
                              GRAPH_FILES[3], "ms", True)

    extension_items = []
    for label, backend, implementation in (
        ("CPU generic extension", "cpu", "generic_extension"),
        ("CPU built-in equivalent", "cpu", "builtin_equivalent"),
        ("GPU generic extension", "gpu", "generic_extension"),
        ("GPU built-in equivalent", "gpu", "builtin_equivalent"),
    ):
        scope = "warm_in_process_host_wall" if backend == "cpu" else "transfer_inclusive_synchronized_host_wall"
        record = select(records, benchmark_name="extension_chain", implementation=implementation,
                        backend=backend, size=1048576, timing_scope=scope)
        extension_items.append((label, ms(record), record, "research_square_linear chain"))
    audit += horizontal_chart(output / GRAPH_FILES[4], "TH-021 extension performance",
                              "Qualified research_square_linear chain · no universal extension-performance claim",
                              extension_items, "Extension and built-in-equivalent medians for the qualified chain.",
                              GRAPH_FILES[4], "ms", False, True)

    deployment_items = []
    for label, backend, implementation, timing_scope in (
        ("CPU warm in-process", "cpu", "thiran_model_runtime", "deployment_warm_in_process_host_wall"),
        ("CPU fresh process", "cpu", "thiran_cli_model_run", "deployment_fresh_process_bundle_load_backend_load_execute_materialize_print_host_wall"),
        ("GPU warm transfer-inclusive", "gpu", "thiran_model_runtime", "deployment_transfer_inclusive_synchronized_host_wall"),
        ("GPU fresh process", "gpu", "thiran_cli_model_run", "deployment_fresh_process_bundle_load_backend_load_execute_materialize_print_host_wall"),
    ):
        record = select(records, benchmark_name="reference_model_deployment", implementation=implementation,
                        backend=backend, size=1, timing_scope=timing_scope)
        deployment_items.append((label, ms(record), record, "reference affine model"))
    audit += horizontal_chart(output / GRAPH_FILES[5], "TH-017 deployment overhead",
                              "Reference affine model · deployment overhead evidence only · log scale",
                              deployment_items, "Warm and fresh-process reference-model deployment medians.",
                              GRAPH_FILES[5], "ms", True)
    return audit


def print_audit(rows: list[AuditRow]) -> None:
    print("| graph | series | size/scope | raw median | rendered median |")
    print("|---|---|---|---:|---:|")
    for row in rows:
        print(f"| {row.graph} | {row.series} | {row.scope} | {row.raw} | {row.rendered} |")


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path,
                        default=repo / "benchmarks/results/TH-023-machine.json")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=repo / "docs/assets/benchmarks/v0.1")
    parser.add_argument("--quiet", action="store_true", help="do not print the value audit")
    args = parser.parse_args()
    try:
        rows = generate(load_records(args.input), args.output_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"benchmark graph generation failed: {error}", file=sys.stderr)
        return 1
    if not args.quiet:
        print_audit(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
