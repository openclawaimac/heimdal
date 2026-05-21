# Core Runtime Requirements

Beta CLI commands:

```bash
heimdal doctor
heimdal run demo
heimdal run --input task.json
heimdal eval run
heimdal patch validate <patch_file>
heimdal logs latest
```

Core modules: intake, role_binding, task_contract, context_os, model_router, quality_factory, verifier, scheduler, storage, repro_trace, patch_manager, hardware_profiler, sandbox.

Required behavior: all tasks become Task Contracts; all model calls get Context Packets; all live outputs pass verification or return need_input/fail; all runs write Repro and Trace Packs; Work Mode preempts Dream/Mirror; Mirror disabled by default.
