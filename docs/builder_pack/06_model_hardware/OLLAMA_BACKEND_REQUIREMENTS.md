# Ollama Backend Requirements

Ollama is the initial model backend. Required: configurable base URL via OLLAMA_HOST, list installed models, generate text, pass parameters, capture latency/metrics if available, timeout/retry policy.

Prefer role-parallelism before model-parallelism.
