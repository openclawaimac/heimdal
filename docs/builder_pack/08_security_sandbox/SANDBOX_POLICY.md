# Capability Sandbox Policy

Default beta policy: read-only access unless task workspace is created, no shell unless explicitly enabled, no network unless tool policy permits it, blocked paths include secrets/system dirs, timeouts on tool calls.

Example blocked paths: ~/.ssh, ~/.openclaw/secrets, ~/.config, /etc.
