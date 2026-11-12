# Gates

A workflow stops and asks the user before any of these. Everything else proceeds.

- Approving a design or a plan
- `git commit`, `git push`, opening or merging a PR
- `sbatch`, `scancel`, rerunning or requeueing a job
- Deleting or moving checkpoints, logs, cache entries or data
- Applying a fix proposed by a security or review finding
- Installing any tool or package
- Editing decks, `seed-note.md`, `state.md`, progress notes, `PLANNING.md`, the kanban
  file, `CLAUDE.md` or memory

## How to ask

- Say what will happen, on which files, branch or job ids, and what it cannot undo.
- Wait for an explicit yes. Silence, or a yes to something else, is not approval.
- One yes covers one action. It does not carry over to the next commit, job or deletion,
  unless the user said so for a stated scope ("commits on this branch, for this plan").
- A workflow that Claude started itself has the same gates as one started by slash command.

## The guard hook is a second line, not the gate

`.claude/hooks/guard_destructive.py` asks or denies on some destructive Bash commands.
It matches command text. It does not see `find -delete`, `rsync --delete`, or a command
inside a script. The gate is this list; the hook is a seatbelt.
