"""Shared status and strictness constants for the Core Runtime.

Centralising these prevents typo-driven drift between the modules that produce
a value (router, verifier) and those that consume it (runtime, eval runner).
"""

# Heimdal Result Envelope / Verification statuses.
PASS = "pass"
FAIL = "fail"
NEED_INPUT = "need_input"

# Verifier strictness levels (produced by the router, consumed by the verifier).
LENIENT = "lenient"
STANDARD = "standard"
STRICT = "strict"
