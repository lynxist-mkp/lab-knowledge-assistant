# Triage labels

Use exactly these label strings when a skill speaks in the five canonical triage roles:

- `needs-triage` -> `needs-triage`: maintainer still needs to evaluate the issue
- `needs-info` -> `needs-info`: waiting on reporter input
- `ready-for-agent` -> `ready-for-agent`: fully specified and ready for an AFK agent
- `ready-for-human` -> `ready-for-human`: requires human implementation
- `wontfix` -> `wontfix`: will not be actioned

Completion check: every triage action should apply the tracker label from the right-hand side, not a paraphrase.
