# Scheduling and partitioning

**IMPLEMENTED:** In the existing normal compiler path, dependency analysis creates execution levels, device planning assigns descriptive devices, partitioning groups Nodes, and communication planning describes crossing values.

**ENGINEERING LIMITATION:** These structures feed normal compiler reporting and BackendIR construction. They do not execute distributed communication and do not control the Region-generated runtime. All current Region strategies execute through PyTorch.

**PLANNED:** Executable device and communication semantics would require a separate verified runtime milestone.

**RESEARCH HYPOTHESIS:** Partitioning or device assignment may eventually improve execution, but no performance result is established.
