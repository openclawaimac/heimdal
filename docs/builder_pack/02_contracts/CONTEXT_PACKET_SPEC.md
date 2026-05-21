# Context Packet Spec

A Context Packet is the exact prompt/context assembly given to an internal model call. It prevents context bloat and makes runs reproducible.

Required sections:

```json
{
  "packet_id": "string",
  "contract_id": "string",
  "role_context": {},
  "truth_context": [],
  "working_state": {},
  "task_context": {},
  "experience_context": [],
  "skills_context": [],
  "budget": {},
  "hashes": {}
}
```

Priorities: task instruction, truth context, working state, role context, selected skills, experience examples.

Do not include all memory. Do not include all skills. Respect token budgets.
