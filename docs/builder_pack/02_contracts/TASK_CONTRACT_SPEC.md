# Task Contract Spec

A Task Contract is created internally by Heimdal for every task.

Required fields:

```json
{
  "contract_id": "string",
  "task_id": "string",
  "role_id": "string",
  "objective": "string",
  "definition_of_done": [],
  "expected_outputs": [],
  "constraints": {},
  "required_sources": [],
  "tool_requirements": [],
  "risk_profile": {},
  "budget": {"quality_level": "B0|B1|B2|B3|B4", "max_iterations": 3, "max_input_tokens": 8000, "max_output_tokens": 2000},
  "verification": {"rubric_id": "string", "requires_citations": false, "requires_schema_validation": false, "no_guess_gate": true}
}
```

Beta rule: every Work Mode task must have a Task Contract before any model call.
