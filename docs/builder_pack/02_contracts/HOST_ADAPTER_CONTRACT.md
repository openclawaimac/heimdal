# Heimdal Host Adapter Contract v0.2

## Host Task Envelope

```json
{
  "host": {"type": "openclaw | hermes | cli | rest | mcp", "host_task_id": "string", "source_agent": "string|null", "callback": {}},
  "role_binding": {},
  "task_request": {},
  "runtime_hints": {}
}
```

## Role Binding

```json
{
  "role_id": "research | ops | dev | finance | general | custom",
  "role_pack": "optional rolepack identifier",
  "risk_mode": "conservative | balanced | aggressive",
  "privacy_mode": "local_only | cloud_allowed",
  "tool_policy": {},
  "memory_scope": {},
  "output_profiles": ["markdown", "json", "files", "code"]
}
```

## Task Request

```json
{
  "task_id": "string",
  "title": "string",
  "instruction": "string",
  "inputs": {},
  "constraints": {},
  "priority": "P0 | P1 | P2 | P3",
  "budget": {},
  "expected_outputs": []
}
```

## Heimdal Result Envelope

```json
{
  "task_id": "string",
  "host_task_id": "string",
  "status": "pass | need_input | fail",
  "message": "string",
  "artifacts": [],
  "questions": [],
  "repro_pack": {},
  "trace_pack": {},
  "metrics": {}
}
```

Required beta adapters: CLI adapter and OpenClaw adapter stub.
