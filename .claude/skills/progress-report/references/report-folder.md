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

## `index.md` template
`| date | title | summary | A | B | C |` one row per report, newest last.

## `INSTRUCTIONS.md` skeleton
Audiences (name, role, cares about, dislikes) / Occasion / Shape / Figures / Sharing / Terms / Rules (copy "Principles that keep reports honest" from the skill).
