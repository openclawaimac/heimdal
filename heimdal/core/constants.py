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

# Verifier backends: rule-based only, or rule-based plus a model-based
# semantic check (the latter is opt-in via the manifest).
RULE_BASED = "rule_based"
HYBRID = "hybrid"
