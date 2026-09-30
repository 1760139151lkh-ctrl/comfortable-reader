---
name: check-book-source
description: Check an answer about this local AI textbook against a published paragraph and its current source hash.
---

Use this workflow only when the question asks what this textbook currently says.

1. Find the relevant published chapter and its explicit paragraph anchor.
2. Compare the chapter bytes with the current local publication entry before reading or citing the paragraph.
3. Read the actual paragraph and the conditions around the claim. Follow `references/source-rules.md` for conflicts and missing evidence.
4. Answer in ordinary language with the chapter and anchor. If the source has changed or does not answer, say what is unknown and stop that claim.

This file describes a workflow. It does not train a model, grant file access, or authorize a tool call.
