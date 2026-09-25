# Command-line interfaces

## Thiran compiler

Exact help text:

```text
Usage:
  Thiran <source-file>
  Thiran --plan <source-file>
  Thiran --emit-plan <source-file> <output-file>
  Thiran --emit-region-executor <source-file> <output-file>
  Thiran --version
  Thiran --help
```

Commands:

```bash
./build/debug/Thiran examples/cnn.th
./build/debug/Thiran --plan examples/cnn.th
./build/debug/Thiran --emit-plan examples/cnn.th /tmp/region-plan.txt
./build/debug/Thiran --emit-region-executor examples/cnn.th /tmp/executor.py
./build/debug/Thiran --help
./build/debug/Thiran --version
./build/debug/Thiran doctor
```

`--version` prints exactly `Thiran 0.2.0-alpha-dev`. `doctor` reports Linux/WSL platform, temporary-directory, Python, Torch, optional CUDA, and `full`, `compiler-only`, or `unusable` mode. Python and Torch are optional for compilation and required only to execute generated Python artifacts.

Normal mode preserves the existing verbose compiler path and writes `generated.py` and `graph.dot` in its working directory. Planning modes stop before scheduling and backend emission and do not modify those default artifacts.

All source modes accept multi-file modules and compile-time tensor functions. Imports resolve relative to the importing file, independently of the current working directory. See [MODULES.md](MODULES.md) and [FUNCTIONS.md](FUNCTIONS.md).

Established driver-level codes are 0 for success, 2 for invalid command usage, 3 for Hybrid Region planning failure, 4 for plan/executor output-file failure, and 5 for Region Python emission failure. Existing frontend and normal-compilation failures retain their historical behavior.

Plan and executor emission refuse normalized source/output identity, overwrite an existing output on success, do not create a missing parent directory, and report deterministic file errors.

## Native artifact runtime

TH-016 installs a separate minimal native runtime/inspector:

```text
thiran-artifact inspect <artifact.tha>
thiran-artifact run <artifact.tha> [--device <ordinal>]
```

`inspect` validates the complete container before reporting its versions,
backend, typed signature, specialization, planning identity, payload, target,
and runtime requirements. `run` is the zero-parameter CLI adapter; parameterized
typed entries use the C++ artifact API. It never invokes the Thiran frontend,
host compiler, PTX generator, Python, or a framework. GPU AOT execution may and
does invoke the CUDA Driver's PTX-to-device-code JIT.

## Native model runtime

TH-017 installs a separate model bundle inspector/runtime:

```text
thiran-model inspect <model.thm>
thiran-model run <model.thm> --backend <cpu|gpu> --input <f32> [--device <ordinal>]
```

The V0 run adapter accepts the reference model's one public
`Tensor<f32,1>[1]` input as a scalar command-line spelling. Learned weight and
bias values are loaded and bound from the bundle; callers never provide them.
Backend selection is mandatory. An unavailable or invalid GPU request fails
explicitly and never falls back to CPU. CPU execution loads the embedded ELF;
GPU execution loads the embedded persistent PTX and permits only the CUDA
Driver's device-load JIT. Neither path invokes the Thiran compiler, a host
compiler, the reference evaluator, Python, or a framework.

## Generated Region executor

Exact help text:

```text
Usage:
  ThiranRegionExecutor --describe
  ThiranRegionExecutor --run <input-bundle.pt> <output-bundle.pt>
  ThiranRegionExecutor --help
```

Commands:

```bash
.venv/bin/python /tmp/executor.py --help
.venv/bin/python /tmp/executor.py --describe
.venv/bin/python /tmp/executor.py --run /tmp/inputs.pt /tmp/outputs.pt
```

The fixed `ThiranRegionExecutor` label is independent of the artifact filename.

Runtime exit codes:

| Code | Meaning |
|---:|---|
| 0 | success |
| 2 | invalid runtime command |
| 3 | input-bundle load or validation failure |
| 4 | Region execution failure |
| 5 | output path or atomic write failure |

## Tensor bundles

Input and output bundles are `dict[str, torch.Tensor]` serialized with `torch.save`. Input loading is restricted:

```python
torch.load(path, map_location='cpu', weights_only=True)
```

The CLI requires exactly the declared inputs. It validates names, Tensor values, rank, fixed dimensions, `-1` wildcard dimensions, and device consistency. It does not retry unrestricted loading. Output tensors are detached, moved to CPU, serialized to a temporary file in the destination directory, and installed with atomic replacement. Input/output path identity is rejected, parents are not created, and an existing output remains unchanged on handled pre-replacement failure.

## Import-safe library use

Importing an artifact performs no execution or file I/O. Its public Python API is:

```python
run_region_plan(inputs)
thiran_input_spec()
thiran_output_names()
```

`run_region_plan(inputs)` accepts a dictionary with all required tensors and permits extra keys for library compatibility. It leaves device placement and returned tensors unchanged. `thiran_input_spec()` and `thiran_output_names()` return immutable tuples.
