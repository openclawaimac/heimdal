# Host-Agnostic Design

Heimdal must not be designed as only an OpenClaw plugin.

Every host integration must translate into a standard Heimdal request:

```text
Host request → Host Adapter → Heimdal Host Task Envelope → Core Runtime
```

Every result must translate back:

```text
Core Runtime → Heimdal Result Envelope → Host Adapter → Host result
```

Initial host types: OpenClaw and CLI. Planned: Hermes Agent, REST API, MCP server, Generic Python SDK.

Adapters are translators. They must not own orchestration logic, build Context Packets, modify patch/eval policies, or bypass sandbox policy.
