# Work, Dream, and Mirror Modes

## Work Mode

Live task execution. Always higher priority than Dream and Mirror.

Budget levels:
- B0: fast direct response, minimal verification.
- B1: worker + verifier.
- B2: worker + retrieval + verifier + repair.
- B3: multi-sample + brain/repair + strict verifier.
- B4: critical mode with adversarial review, tests, schema validation, and extended trace.

## Dream Mode

Background improvement when machine is idle. Generates synthetic tasks, reruns failed tasks, mines failures, proposes patches, and generates eval cases. Never auto-merges to stable.

## Mirror Mode

Optional cloud teacher comparison. Disabled by default. Has token/cost caps, privacy checks, and logs all external calls. Produces patch proposals only.
