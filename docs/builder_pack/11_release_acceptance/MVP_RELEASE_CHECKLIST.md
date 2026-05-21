# MVP Release Checklist

Commands:

```bash
heimdal doctor
heimdal run demo
heimdal run --input examples/tasks/simple_task.json
heimdal eval run
```

Expected files: repro packs, trace packs, hardware profiles, eval summary.

Manual checks: does Heimdal return need_input instead of guessing? Does it run without Ollama? Does it detect WSL2? Is OpenClaw separate from core?
