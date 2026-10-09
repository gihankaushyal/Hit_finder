# Hit_finder dashboard

A local web dashboard for this repository: the latest pull request, open issues,
the kanban board, and CI and local test status. It runs on Sol, binds to
`127.0.0.1` only, and is opened from a laptop through an SSH tunnel.

Milestone 1 (this package) covers the status panels and the kanban. Agent
orchestration, the chief-of-staff chat and the terminal arrive in later
milestones; their regions in the UI are labelled placeholders.

Nothing here is imported by `src/`, and the Python test suite does not depend on it.

## Run it

Requires Node 24 and an authenticated `gh` (`gh auth status`).

```bash
cd dashboard
npm ci
npm run build
npm start
```

`npm start` prints one URL of the form `http://127.0.0.1:4317/?token=...`.
On the laptop, forward the port and open that URL:

```bash
ssh -L 4317:127.0.0.1:4317 <you>@<sol-node>
```

The first visit sets a cookie and removes the token from the address bar.

### Why there is a token

Compute nodes are shared, so other users on the same node can reach a localhost
port. Every request (pages, API, event stream) must carry the token. It is
generated on first start and stored in `dashboard/.state/token` with mode `0600`.
Delete that file to rotate it. Do not paste the tokenised URL anywhere shared.

## Kanban sync

Each task is a GitHub issue labelled `task`, plus one of `status:todo`,
`status:in-progress`, `status:blocked` (a closed issue is Done) and a `kind:`
label taken from the markdown section. `phase-05-kanban.md` stays the file you
edit by hand; each item is linked to its issue by a `<!-- gh:#N -->` comment on
its first line.

Only a `- [ ]` or `- [x]` at the start of a line is a task. Indented checkboxes
belong to their parent item.

### First import (manual, once)

```bash
npm run kanban:sync -- --dry-run   # prints what would happen; writes nothing anywhere
npm run kanban:sync -- --yes       # creates the issues and writes the markers
```

The real run creates one issue per item (ticked items are created, then closed),
pausing between creates, and copies the file to `phase-05-kanban.md.pre-sync.bak`
before its first change. Do not edit the file while it runs: the sync stops
rather than overwrite an edit. If it is interrupted, run the same command again;
it links to issues it already created instead of creating them twice.

Until an import has completed, the board is read-only and the server does no
syncing. The server never performs the first import by itself.

### After the import

The server syncs when the file changes, when you change a task in the UI, and
every 60 s.

| Change | Result |
|---|---|
| New unmarked item in the file | Issue created, marker written |
| Box ticked or unticked in the file | Issue closed or reopened |
| Issue closed or reopened on GitHub | Box ticked or unticked |
| Item text edited | Issue body updated |
| Task added in the UI or on GitHub | Appended under `## Inbox (added via dashboard)` |
| Item removed from the file | Issue left alone, shown as "not in file" |

Nothing is ever deleted. A background sync creates at most 5 issues in one run;
above that it asks you to run the command yourself.

Other flags: `--rebuild-state` rebuilds the sync record from the file and GitHub
when `.state/kanban-sync.json` was lost (it changes no boxes and no issues).

## Tests panel

"Run tests" runs `pytest tests/ -q` with the interpreter in `DASH_PYTHON` and
reads the junit report from `.state/junit.xml`. One run at a time; the run is
stopped if the server shuts down.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DASH_PORT` | `4317` | Port on `127.0.0.1` |
| `DASH_REPO_ROOT` | parent of `dashboard/` | Repository the panels describe |
| `DASH_STATE_DIR` | `dashboard/.state` | Token, junit report, sync state and lock |
| `DASH_KANBAN` | `<repo>/phase-05-kanban.md` | Kanban markdown file |
| `DASH_GH_BIN` | `gh` | GitHub CLI binary |
| `DASH_PYTHON` | `python` | Interpreter used for the local test run |

## Development

```bash
npm run dev:server   # API with reload
npm run dev:web      # Vite dev server
npm test             # unit and API tests (Vitest; server and web projects)
npm run typecheck
npm run e2e          # Playwright against the built app, with a fake gh and fake pytest
```

The end-to-end harness refuses to start unless every path points into a temp
directory and `gh` is the fake, so it cannot touch the real kanban file or the
real repository. If Playwright's bundled browser is missing, set
`E2E_CHROMIUM_PATH` to a Chromium binary.

Layout: `server/` (Hono API; `lib/kanbanMd.ts` is the byte-preserving markdown
parser, `lib/kanbanSync.ts` the sync), `web/src/` (React panels, plain CSS with
tokens in `tokens.css`), `shared/types.ts` (API types), `tests/`, `e2e/`.
All subprocesses (`gh`, `git`, `pytest`) are run with argument arrays, never
through a shell.
