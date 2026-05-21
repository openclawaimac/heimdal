# Storage Layout

Required default layout:

```text
storage/
  truth/
  working_state/
  experience/
  skills/
  patches/stable/
  patches/beta/
  patches/experimental/
  patches/rejected/
  eval/
  eval_runs/
  artifacts/
  logs/repro_packs/
  logs/trace_packs/
  logs/hardware_profiles/
  workspace/
```

WSL2 rule: store this inside the Linux filesystem, not /mnt/c or /mnt/d.
