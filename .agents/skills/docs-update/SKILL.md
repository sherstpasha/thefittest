---
name: docs-update
description: Update documentation and examples to reflect implemented behavior, code changes or audit findings. Use when documentation edits are requested or included in an authorized implementation task.
---

# Update documentation

## Establish the required changes

Read applicable repository instructions and inspect existing edits. Identify the
requested scope and locate authoritative documentation sources, generated output
and build tooling. Follow the project's organization, formats and languages
instead of introducing another documentation system.

Use supplied audit findings as the starting point and verify their evidence.
Investigate missing context without repeating a full audit unnecessarily.
If a mismatch may be an implementation defect, establish the intended contract;
do not change runtime behavior or document a suspected bug as intended merely
to remove the discrepancy. Continue independent confirmed corrections.

## Update the affected sources

Keep guides, API documentation and examples consistent with the public interface.
Explain user-visible behavior, defaults, constraints and results. Add navigation
or migration guidance where needed. Document the component's stable contract,
not temporary implementation status or only its most recently added option.

Match the requested information hierarchy and level of detail. Prefer an
existing destination over a new page or section unless a separate destination
is requested or necessary. Moving content includes removing obsolete containers
and redundant references from its former location.

Keep reader-facing prose focused on information needed to use the documented
feature. Omit self-evident introductions, descriptions of the page itself and
maintenance or implementation narration. Do not add interface elements or labels
as incidental extras; include them only when requested or needed for the task.

Keep change logs concise and parallel, with one concrete change per entry.
Follow the requested format rather than inventing categories or extra commentary.

Preserve unrelated content and user changes. Edit authoritative sources rather
than patching generated pages as a substitute. Regenerate tracked output when
it is part of the requested update and the repository's workflow.

If historical versions or archiving are in scope, preserve their provenance and
content separately from current-release changes. Historical API generation must
use its matching implementation, not silently import the current code.

## Validate and finish

Use checks appropriate to the edits: documentation builds, rendered sections,
links, API references and small executable examples. Rendering does not execute
examples or prove API coverage; inspect the affected output as needed.
Reuse completed checks while their inputs remain unchanged, and rerun affected
checks after further edits.

Fix defects introduced by the update. Distinguish pre-existing problems and
environmental blockers from introduced failures, and report unverified checks
honestly rather than suppressing warnings to obtain a successful build.

Review the final diff for unrelated edits, generated-file churn and lost user
work. Report what changed, validation results and remaining limitations.
Editing or building documentation does not itself publish it.
