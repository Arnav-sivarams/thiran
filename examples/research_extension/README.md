# Research extension example

This trusted native compiler plugin declares one rank-1/rank-2 f32 operation,
its scalar recipe, CPU/GPU availability, fusion eligibility, and reverse rule.
Build it from the repository root:

```bash
c++ -std=c++20 -fPIC -shared -Iinclude \
  examples/research_extension/square_linear_extension.cpp \
  -o /tmp/thiran-square-linear.so

build/release/thiran check examples/research_extension/square_linear.th \
  --extension /tmp/thiran-square-linear.so
build/release/thiran run examples/research_extension/square_linear.th \
  --backend cpu --extension /tmp/thiran-square-linear.so
build/release/thiran build examples/research_extension/square_linear.th \
  --backend cpu --extension /tmp/thiran-square-linear.so \
  -o /tmp/square-linear.tha
build/release/thiran artifact run /tmp/square-linear.tha
```

The last command does not load the extension `.so`: validated scalar recipes
are embedded into `.tha` artifacts. Loading an extension is still a trusted
native-code operation; V0.1 provides no sandbox or signature verification.
