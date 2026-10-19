# Phase 5 Kanban — Staged To-Do Items

> Local-disk-only working list.

---

## Awaiting merge (PRs open)

_(empty — nothing
open)_

## Open decisions (need a call)

- [x] **Re-pretrain SSL folds 1–2?** → **Decided 2026-10-05: all 4 folds**
      will be re-pretrained.
- [ ] **crops_per_frame>1 degrades val AP** — 2026-10-05, smoketest:

      | crops | AP |
      |---|---|
      | 1 | 0.9464 |

      Train loss drops monotonically.

## In progress (Phase 5)

- [ ] Risk B4 — diagnose early stopping (lr 5e-5 peaks
      within ~2 epochs)
- [x] Frame cache build-out <!-- gh:#12 -->

## All tracked tasks (snapshot)

### Group A
- [x] Task 0: worktree setup
- [ ] Fold-2 A/B validation of frame cache
  - [ ] nested sub-bullet, not a task item
  - plain sub-bullet <!-- gh:#99 -->

### Group B
* [ ] star bullet is not a task item

---

## Done (kept briefly)

_(empty)_
