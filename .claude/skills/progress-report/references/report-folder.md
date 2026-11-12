# `reports/` folder (local, gitignored)

```
reports/
  INSTRUCTIONS.md      audiences, occasion, tone, figure and sharing rules (from the first-run interview)
  index.md             running log: | date | title | one-line summary | deck A/B/C links |
  glossary.md          plain-language definitions reused every report
  YYYY-MM-DD/   (or YYYY-MM-DD-2 for a further report on the same day)
    facts.md           verified claims table + "Since last report" status block + "not verified" list
    audience-A.md  audience-B.md  audience-C.md
    deck-A.md  deck-B.md  deck-C.md        slide content, speaker notes, artifact URL
    figures/
    state.md           continuation record
```

## `facts.md` columns
`id | claim | value | source (file:line / run name / PR) | date | confidence (seeds, folds finished, cross vs in-domain) | used in (A/B/C)`

## `state.md` template
```
# State after <date>
## Claimed this report
- <claim ids and one-line headline per audience>
## Metrics baseline (copy exact values; next report compares against these)
## Open questions
## Promises for the next report
- [ ] <promise> (owner, rough timing)
## Dropped / changed since the previous report
```

## `docs/progress_notes.md` section template (appended by step 6)
```
## N. <title> (<period start> … <period end>)
Report <date> (`reports/<date>/`, local) carries the verified claims; ids below point into its `facts.md`.
### N.1 What was done / found   (one subsection per topic; numbers recomputed <date>, sample sd n-1, seeds and folds stated)
### N.x Corrections to earlier sections   (say what changed and which section it supersedes)
### N.y Next   (open promises and pending user decisions)
```
The heading must start with the number and carry a date or date range: `collect_since.py` finds sections by the heading date.

## `index.md` template
`| date | title | summary | A | B | C |` one row per report, newest last.

## `INSTRUCTIONS.md` skeleton
Audiences (name, role, cares about, dislikes) / Occasion / Shape / Figures / Sharing / Terms / Rules (copy "Principles that keep reports honest" from the skill).
