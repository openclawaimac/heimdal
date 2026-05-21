# Acceptance Test Catalog

Must-pass tests:
1. CLI doctor runs without Ollama and exits cleanly with warning.
2. CLI doctor runs with Ollama and lists models.
3. Demo run writes Repro Pack.
4. Demo run writes Trace Pack.
5. Task Contract is created before model call.
6. Context Packet is created before model call.
7. Verifier returns structured JSON.
8. No-Guess Gate blocks unsourced factual claim when sources required.
9. Patch validation rejects invalid patch schema.
10. Eval runner produces summary JSON.
