---
name: docs-audit
description: Check documentation accuracy and completeness against implementation, public APIs and requested changes. Use for documentation audits and freshness reviews; report findings without editing unless updates are requested.
---

# Audit documentation

## Establish scope

Read applicable repository instructions. Identify the requested feature, change
set, release or whole-project scope. Locate documentation sources, generated
output and build configuration from the repository rather than assuming paths,
formats, languages or tools.

Use an explicit baseline when supplied. Otherwise inspect current behavior and
relevant history, stating the coverage limits. A whole-project audit must also
consider existing undocumented behavior, not just recent changes.

## Compare documentation with behavior

Trace user-facing behavior through implementation, public exports, tests and
package metadata. Check installation, dependencies, signatures, defaults,
results, constraints, errors and examples as relevant to the scope.

Identify incorrect claims, missing guidance, stale examples and broken links.
Check whether users can discover and apply a feature, not merely find its name
in generated API pages. If code contradicts an established contract, report a
possible implementation defect instead of assuming the docs must adopt it.

Treat historical documentation according to its release. If versions or archives
are in scope, verify their provenance and navigation without expecting old pages
to describe current behavior.

## Verify and report

Choose checks appropriate to the affected surfaces: builds, rendered pages,
API references, links and small executable examples. Use the project's tooling.
Keep review-only work from modifying tracked sources or published artifacts.
A successful build does not prove complete or accurate content.

Report actionable findings in order of user impact, with locations, supporting
evidence and suggested corrections. Separate confirmed defects from suspicions
and unverified areas. State the reviewed scope, checks performed and limitations.
If updates are also authorized, use these findings for the update workflow;
otherwise leave the repository unchanged.
