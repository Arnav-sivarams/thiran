# Modules

The file passed to Thiran is the entry module. Imports must precede assignments and use exactly:

```thiran
import "weights.th" as weights
```

Paths are relative to the importing file, must end in `.th`, and must resolve to a regular file within the canonical parent directory of the entry file. There is no current-working-directory fallback, package lookup, environment search path, standard library, or remote import.

Imported modules expose values explicitly:

```thiran
export W = Constant(1, 2, 2)
```

An importer uses `weights.W`. Unexported values remain local. Aliases are mandatory. Cycles, root escapes, duplicate aliases, unresolved references, and access to unexported values are errors. Only the entry module may contain `Input` or `Output`.

Modules are discovered deterministically in import source order. Each canonical file is loaded once, imported modules are lowered before importers, and the result is one ordinary flattened Graph. Imported Node names use a deterministic internal prefix and contain no absolute paths.

Diagnostics use one-based line and byte-column positions. A tab advances the reported column by one byte; LF and CRLF both advance to the next line at the LF byte.

Modules may export top-level tensor functions with `export fn`; see [FUNCTIONS.md](FUNCTIONS.md). Thiran does not provide classes, packages, re-exports, wildcard imports, native object linking, or multiple executable Graphs.
