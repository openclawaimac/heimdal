# Adapter Layer Requirements

Required beta adapters: CLI adapter and OpenClaw adapter stub.

Each adapter implements `to_host_task_envelope(raw_input)` and `from_heimdal_result(result)`. Adapters translate; they do not orchestrate.
