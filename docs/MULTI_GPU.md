# Multi-GPU and multi-machine routing

Heimdal runs each internal role — worker drafting, semantic verification,
brain planning, coding — against a *backend*. By default there is one
backend and every role shares it. From v0.7.0 the manifest can map roles
onto separate Ollama endpoints, so a machine with more than one GPU (or a
house with more than one machine) actually gets used.

This is **role-level routing**. It is not tensor or pipeline parallelism:
splitting a single model's weights across devices remains Ollama's and
llama.cpp's job, and Heimdal never tries to do it.

## One Ollama per GPU

Start an Ollama instance per device, each pinned to its own GPU and port:

```bash
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ollama serve
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

Then map roles to them:

```yaml
ollama:
  base_url: http://localhost:11434
  endpoints:
    - name: gpu0
      base_url: http://localhost:11434
      roles: [worker, brain]
    - name: gpu1
      base_url: http://localhost:11435
      roles: [semantic_verifier]
```

A role with no endpoint falls back to `base_url`. With `endpoints: []` —
the shipped default — behaviour is identical to pre-v0.7.0.

Check the wiring before relying on it:

```bash
heimdal endpoints list      # role -> endpoint name
heimdal endpoints status    # reachability + models present per endpoint
```

The resolved routing is recorded in every Trace Pack as an
`endpoint_routing` event, so a run can be audited after the fact.

## Slots: endpoints that are themselves routers

Some endpoints are not a single GPU but a proxy that fans requests out
over several machines. [NVIDIA PAIR](https://github.com/NVIDIA/Personal-AI-Router)
is the notable one: it discovers paired nodes on the LAN over mDNS,
schedules on queue depth and GPU utilisation, and presents a single
Ollama-compatible endpoint. From Heimdal's side that is one `base_url`
that can absorb many concurrent requests.

Declare that width with `slots`:

```yaml
ollama:
  endpoints:
    - name: pair
      base_url: http://127.0.0.1:11434
      roles: [worker, brain, semantic_verifier]
      slots: 3          # three paired nodes behind the router
```

`slots` defaults to 1, so existing configs are unaffected. A value below 1
or a non-integer is clamped to 1.

## Concurrent sampling

At quality levels B3 and B4 the worker drafts several candidate samples.
Whether those run concurrently is governed by `concurrency.parallel_samples`:

| value | behaviour |
| --- | --- |
| `auto` (default) | parallel when the worker role has more than one slot |
| `true` | always parallel when samples > 1 |
| `false` | never; serial, pre-v0.7.0 behaviour |

Under `auto`, both shapes qualify: several single-slot endpoints, or one
multi-slot router endpoint. Counting *distinct endpoints* would have
missed the router case — one `base_url` fronting three machines would
have drafted serially and left the cluster idle.

When parallel sampling engages, the Trace Pack records a `parallel_samples`
event with the slot count, so you can confirm the width a run actually used.

## Health-aware failover

Pinning a role to one endpoint makes that endpoint a single point of
failure. If the semantic verifier is mapped to GPU 1 and GPU 1 stops
answering, the run should not die while GPU 0 sits idle and able to serve
it. So each role has an ordered list of candidate endpoints, and a request
that fails on one is reissued on the next.

`ollama.failover` picks the policy:

| value | fallback order for a role |
| --- | --- |
| `auto` (default) | the role's own endpoints, then any other endpoint in the pool as a borrowed spare |
| `strict` | only the role's own endpoints |
| `off` | none; a downed endpoint fails the run |

Under `auto`, running the verifier on the worker's GPU is worse than
running it on its own — but far better than failing the run. Under
`strict` you keep the isolation and accept the failure. `off` restores
pre-v0.7.1 behaviour.

`heimdal endpoints list` prints the resolved chain per role:

```
  failover: auto
    worker             gpu0 -> gpu1
    semantic_verifier  gpu1 -> gpu0
```

### Circuit breaker

An endpoint that raises is taken out of rotation rather than retried on
every subsequent request. After `failover_cooldown_seconds` (default 60)
one trial request is allowed through; if it succeeds the endpoint is back
in rotation, if it fails the endpoint drops out again. The ledger is
session-scoped and shared across roles, so one role discovering a dead
GPU spares the others from rediscovering it.

Note that the per-endpoint retries `ollama.max_retries` configures happen
*first*, inside a single endpoint. Failover only engages once an endpoint
has exhausted those — a single failover therefore means the endpoint
genuinely failed several times, which is why one failure is enough to open
its circuit.

If every candidate's circuit is open, requests are attempted anyway rather
than refused outright: the cooldown may simply not have elapsed on an
endpoint that has since recovered.

### What gets rerouted

Connection failures, timeouts, and missing-model (HTTP 404) responses all
fail over — another endpoint may well satisfy any of them. Errors from
anywhere else in the pipeline are not caught, so a genuine bug surfaces
instead of being masked by a retry storm across the cluster.

### Observability

A degraded run is visible rather than just slower:

- `endpoint_failover_policy` (Trace Pack) records the candidate chain at
  the start of a run.
- `endpoint_unhealthy` records each endpoint that failed, with the error.
- `endpoint_failover` records which endpoint ultimately served the
  request, which ones were tried and failed, which were skipped for
  having an open circuit, and whether the winner was borrowed from
  another role.
- `metrics.endpoint_failovers` counts every request in *this run* that did
  not land on its configured first choice — including ones where the first
  choice was skipped rather than tried. `0` means the run was not degraded
  at all. A host is encouraged to reuse one `Runtime` across many tasks, so
  this is reported as a per-run delta; the circuit-breaker ledger behind it
  is deliberately session-scoped, and `metrics.endpoint_health` (present on
  a failed run) is that cumulative view.

## What this does not do

- It does not pool GPU memory or let a model larger than one device run.
- It does not shard a model or split an in-flight request across devices.
- It does not resume a partially generated response on another endpoint;
  a failed request is reissued from the start.
- It does not health-check endpoints ahead of time. The first request to
  a dead endpoint is what discovers it — `heimdal endpoints status` pings
  them on demand, but a run does not.
