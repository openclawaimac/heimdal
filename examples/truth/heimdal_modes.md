# Heimdal Operating Modes

Heimdal Engine runs in three operating modes. Work Mode executes live tasks and
always has higher priority than the background modes. Dream Mode performs
background self-improvement when the machine is idle, generating synthetic
tasks, mining failures, and proposing patches, but it never merges changes into
the stable channel by itself. Mirror Mode optionally compares Heimdal outputs
against a cloud teacher model, is disabled by default, and only produces patch
proposals under strict token and privacy limits.
