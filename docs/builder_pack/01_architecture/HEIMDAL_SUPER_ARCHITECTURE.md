# Heimdal Super Architecture

## External View

To any host framework, Heimdal appears as one agent:

```text
Host CEO / Planner / User → Heimdal Agent → Result + Artifacts + Status + Repro/Trace pointers
```

## Internal View

```text
Heimdal Engine
├─ Host Adapter Layer
│  ├─ OpenClaw adapter
│  ├─ Hermes adapter
│  ├─ CLI adapter
│  ├─ REST adapter
│  └─ MCP adapter
├─ Core Runtime
│  ├─ Intake
│  ├─ Role Binding Resolver
│  ├─ Task Contract Builder
│  ├─ Context OS
│  ├─ Model Router
│  ├─ Quality Factory
│  ├─ Scheduler
│  ├─ Patch Manager
│  ├─ Eval Runner
│  ├─ Trace/Repro Logger
│  └─ Sandbox Controller
├─ Modes
│  ├─ Work Mode
│  ├─ Dream Mode
│  └─ Mirror Mode
└─ Storage
   ├─ Truth Vault
   ├─ Working State
   ├─ Experience Graph
   ├─ Skills
   ├─ Patches
   ├─ Evals
   ├─ Artifacts
   └─ Logs
```

## Non-negotiable Kernel Components

Beta must include Host Adapter Contract, Task Contract, Context Packet, Repro Pack, Trace Pack, Verifier PASS/FAIL, No-Guess Gate, Hardware/Profile Doctor, Patch format, minimal eval gate, and Sandbox policy.
