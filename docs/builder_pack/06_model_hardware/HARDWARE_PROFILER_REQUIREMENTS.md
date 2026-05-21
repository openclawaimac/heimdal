# Hardware Profiler Requirements

`heimdal doctor` must detect OS, native Linux vs WSL2, CPU, RAM, disk class, Ollama reachability, installed Ollama models, GPU count, VRAM per GPU, CUDA availability, and Apple Silicon/Metal if relevant.

Deployment modes: Dev, Single Device, Pipeline, Factory. Profiler writes `storage/logs/hardware_profiles/<timestamp>.json`.
