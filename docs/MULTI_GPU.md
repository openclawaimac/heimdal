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

## What this does not do

- It does not pool GPU memory or let a model larger than one device run.
- It does not shard a model or split an in-flight request across devices.
- It does not retry a failed sample on a different endpoint; endpoint
  health is reported by `heimdal endpoints status`, not routed around
  mid-run.
