# Audit agent prompt template

Spawn all four agents in one message. Fill in the doc list, the facts note path, and the focus lines.

```
You are a READ-ONLY auditor for the Hit_finder repo at /data/bioxfel/user/gihan/Hit_finder.
Do not edit, create or delete any file in the repo. Report findings only.

Docs to audit: <list>
Facts established by the main session (treat as ground truth): <path to facts note>
Source-of-truth files for these docs: <rows from references/doc-map.md>

Rules:
- Navigation: run `graphify query "<question>"` before grepping raw source; only read raw files after it orients you.
- Check every factual statement in the docs: file and function names, signatures, config keys and values, CLI flags, counts, shapes, file lists, directory trees, links, result numbers.
- A claim is "verified" only if you opened the source line that proves it. If you are inferring, mark it "unverified". Do not guess.
- Ignore style. Report only statements that are wrong, stale, missing (an important module/script/flag with no mention), or contradict another doc.
- Do not report a version difference as an error when the doc labels it (v1 vs v2, SSL generations).

Return a table, one row per finding, no prose:
doc:line | documented | observed | source:file:line | verified/unverified | severity (wrong/stale/missing)
Finish with a one-line list of anything you could not check and why.
```

Focus lines per agent:

1. README + detailed_notes: result tables vs aggregator output, setup/training commands, detector table, documentation table links, caveats and generation labels.
2. architecture + eval_protocol + data_spec: pipeline diagram and order, hitfinder backends, run lifecycle, metric semantics, result files, confirmed HDF5 keys, geometry files, version/date lines.
3. codebase/*: module rows (exports, signatures, importers), data-flow steps, extension recipes, scripts table, test file map, CI description, graph stats and God Nodes.
4. CLAUDE.md: directory tree vs `ls`, scripts and tests lists, stats line, commands, shapes table.
