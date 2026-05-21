# OpenClaw Adapter Spec

OpenClaw sees Heimdal as one agent. Heimdal may be assigned any role in an OpenClaw setup. Heimdal internally resolves the role pack and pipeline.

The OpenClaw adapter must accept task payloads from OpenClaw-like input, map them into the Heimdal Host Task Envelope, return a Heimdal Result Envelope, and store Repro Pack and Trace Pack.

OpenClaw should not directly address Heimdal Router, Worker, Verifier, Dreamer, etc.
