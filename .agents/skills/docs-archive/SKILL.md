---
name: docs-archive
description: Preserve a documentation release or existing site as a self-contained historical snapshot before further updates. Use when documentation archiving is requested; retain provenance and verify that the archived site remains usable.
---

# Archive documentation

## Identify the snapshot

Read applicable repository instructions and inspect existing archive and build
conventions. Establish which documentation the user wants to preserve: an
existing rendered site or a specific historical release. Locate the entry page,
sources, assets and version metadata before choosing the archive label.

Do not infer a documentation version from today's package metadata. If labels,
history and the requested baseline conflict, clarify the identity before making
a version claim. A dated or descriptive snapshot label is appropriate when the
user wants the current state rather than a verified release.

## Preserve the content

Follow the existing archive layout where available. Copy a rendered snapshot
with its pages, relative directory structure, assets, search data and other
required resources. Do not rebuild it against the current implementation or
silently update its examples, defaults or API descriptions.

If a historical build is necessary, use matching documentation and package
sources in an isolated location. Record the source revision, original location,
displayed version, languages and any known provenance limitations alongside
the snapshot. Distinguish a working-tree snapshot from a release build.

Keep active documentation and user edits intact until the archive is verified.
Do not overwrite an existing archive silently. Exclude caches, environments,
temporary builds and unrelated project files. Produce a downloadable package
only when requested or required by the project's archive workflow.

## Verify and expose the archive

Compare copied content with the original, using hashes where useful. Check the
entry page, local links, scripts, styles, images and search resources. Separate
pre-existing broken links from damage caused by relocation.

Expose the snapshot through the project's existing navigation when in scope.
Keep navigation-only changes separate from frozen content. Verify links in both
directions; pages or anchors absent in another version need a valid destination.
For a downloadable archive, also test extraction and opening the entry page.

Report the archive location, label, provenance, checks and limitations. Do not
claim publication or begin updating the next release unless that work is also
authorized for the current iteration.
