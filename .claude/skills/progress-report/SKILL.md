---
name: progress-report
description: Produce the Hit_finder stakeholder progress report as three audience-tailored write-ups and slide decks (domain PI with little AI background; domain expert with moderate AI knowledge; AI expert with light domain knowledge), each continuing the previous report. Builds a verified fact base from docs/progress_notes.md, re-aggregated results, merged PRs and the kanban, then publishes decks as private Slides artifacts. On the first run it interviews the user and sets up the local, gitignored reports/ folder. Manual only: run /progress-report.
disable-model-invocation: true
---

# progress-report

Turn the project's recent work into reports that three different PIs can each act on, and make each report pick up where the last one stopped. The risk in this job is not writing; it is saying something about results that is not true, or telling the same story to people who need different stories. So the workflow is built on two ideas: **one verified fact base, rendered three ways**, and **every report starts from the previous report's state**.

Read `references/audiences.md` before writing anything and `references/known-pitfalls.md` before touching numbers.

## Where things live

- Skill (tracked): `.claude/skills/progress-report/`.
- Reports (local only, gitignored): `reports/` at the repo root. Layout and file templates are in `references/report-folder.md`. Tracked docs and code must never link into `reports/`.

## Workflow

Run everything from the **main checkout** (`/data/bioxfel/user/gihan/Hit_finder`). `reports/`, `docs/progress_notes.md`, `phase-05-kanban.md` and `PLANNING.md` are gitignored, so a worktree does not have them; there the skill would think it is the first run and build an empty digest. The script moves to the main checkout itself; do the same for any other command.

### 0. First run: set up the folder
If `reports/INSTRUCTIONS.md` does not exist, this is the first run. Do not guess the stakeholders. Run `superpowers:brainstorming` and use `AskUserQuestion` following `references/first-run-interview.md` until the answers cover roughly 95% of what you need (who the three audiences are, meeting format, tone, length, figures, terms to avoid, whether links may be shared). Then create `reports/INSTRUCTIONS.md`, `reports/glossary.md` and `reports/index.md` from the templates in `references/report-folder.md`, and check `git check-ignore reports/x`. If `reports/` is not ignored, tell the user the one line to add to `.gitignore` and ask before editing it (their `.gitignore` often has unrelated uncommitted edits; stage nothing else).

Later runs skip the interview. Re-ask only if the user says an audience, meeting or preference changed, and update `INSTRUCTIONS.md` when they do.

### 1. Preflight
```bash
python .claude/skills/progress-report/scripts/collect_since.py --reports-dir reports
```
It prints the latest report folder, its `state.md` and the period start. Read `reports/INSTRUCTIONS.md`, `reports/index.md` and the latest `state.md` first. The new report covers everything since the previous report's date; if there is none, ask the user how far back the first report should reach. Confirm the period with the user in one line. The baseline is the newest report folder that has a `state.md`; an unfinished folder is flagged and skipped: ask the user whether to resume it (its `facts.md` may hold verified claims) or delete it, rather than silently starting `-2` beside it. Folders are `YYYY-MM-DD`, or `YYYY-MM-DD-2`, `-3` for a further report on the same day: never overwrite an existing report folder.

### 2. Gather (read-only, in parallel)
```bash
python .claude/skills/progress-report/scripts/collect_since.py --since YYYY-MM-DD --full
```
gives the progress-note sections, merged PRs, commits and kanban/plan state for the period. Then re-aggregate every result you intend to quote: `python scripts/aggregate_lodo_results.py --select all` (sample std, generations kept apart; see pitfalls), and W&B summaries where the notes lack a number. Notes are leads; the aggregator and W&B are the source for figures.

### 3. Build `facts.md`
Create `reports/YYYY-MM-DD/facts.md`: one row per claim with value, source (file/line, run name, PR), date, and a confidence tag (seeds, folds finished, in-domain vs cross-detector). Re-check each claim against its source yourself before it goes in. Drop what you cannot confirm and list it under "not verified". Add a "Since last report" block: for every promise in the previous `state.md`, mark done, changed, dropped or still open, with the evidence.

### 4. Agree the message before building anything
Show the user: the claims table, the one-sentence headline per audience, the promised-vs-delivered list, the draft progress-notes section (step 5), and what is deliberately left out. Wait for edits. Building three decks on a wrong headline is the expensive failure; this step is cheap.

### 5. Update the project log
Once the claims are agreed (step 4), update `docs/progress_notes.md` before building anything, so the lab record is current even if the session ends mid-build and so the notes, the write-ups and the decks all start from the same approved facts (it is the record the next run's digest and later sessions read first). Draft one new numbered, dated section `## N. <title> (<period start> … <period end>)` from the verified `facts.md` using the template in `references/report-folder.md` (the digest prints the next number). Show the draft to the user with the claims table in step 4, and append it on their go-ahead. **Append only:** do not rewrite earlier sections; when this report supersedes one (an experiment that was open, a result that changed), say so in the new section's "Corrections to earlier sections" and add one `> Superseded in part by section N` line directly under the heading of the old section or subsection (never above a `## ` heading: the digest would attach it to the previous section). If building the write-ups or decks later turns up a wrong number, append the correction to this same section. No job IDs. The file is local and gitignored, so there is nothing to commit.

### 6. Write
Write the three write-ups first (`audience-A.md`, `-B.md`, `-C.md`, one to two pages each): they are the source of truth. Then derive the three decks from them using the slide spine in `references/slide-spine.md`. Audiences differ in vocabulary, depth and which result leads, never in the numbers. Make figures with matplotlib into `figures/` (label seeds and error bars; no truncated axes that exaggerate).

Build each deck as a Slides artifact: `Artifact action:"quickstart" intent:"slides"`, list and use the user's default design system, publish with the type's `type_url`, fill it, then view it. Save the slide content as `deck-A.md` etc. next to the write-up with the artifact URL. Artifacts are private by default; sharing them or exporting to .pptx/PDF for anyone else is the user's decision, so offer it, do not do it.

### 7. Close the loop
Write `state.md` (what this report claimed, open questions, promises for next time, the metrics baseline) and add a line to `reports/index.md`.

Report to the user: links, files, claims dropped, items you could not verify, and open promises. Stop there; do not publish, share, push or edit the kanban without their go-ahead.

## Principles that keep reports honest
- Single-seed results are labelled as single-seed; unresolved or negative findings appear (a regression is information for the AI-expert audience and context for the domain PI).
- A metric a lay audience will see gets a plain-language meaning from `glossary.md` once, then is used consistently.
- Continuity beats novelty: if last report promised something, this report says what happened to it, even when the answer is "dropped, because...".
