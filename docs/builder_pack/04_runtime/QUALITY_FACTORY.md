# Quality Factory

Pipeline: Task Request → Task Contract Builder → Context Builder → Router → Worker Draft → Verifier PASS/FAIL → Repair Loop if FAIL → Final Result → Repro + Trace logging.

Router selects budget, model profile, tool plan, retrieval requirement, verifier strictness, and whether Brain/Coder is needed.

Verifier returns structured JSON and must not rewrite the answer. Repair loop defaults to max 2 iterations.
