# Hit_finder dashboard

A local web dashboard for this repository: the latest pull request, open issues,
the kanban board, and CI and local test status. It runs on Sol, binds to
`127.0.0.1` only, and is opened from a laptop through an SSH tunnel.

This package covers the status panels, the kanban, the training-runs table,
worker agents with approval cards, the chief-of-staff chat and a terminal.

Nothing here is imported by `src/`, and the Python test suite does not depend on it.

## Layout

Three tabs, kept in the address (`?tab=kanban`, `?tab=runs`; Overview has no `tab` parameter, so
every tab can be bookmarked or opened in a new window). Runs filters and sort (`kind`, `archived`, `sort`,
`dir`) live in the same address and carry across tabs, so a filter set on Runs is still applied when you
return to Overview:

- **Overview**: Latest pull request, Tests (GitHub Actions and the local run) and a compact Kanban across the top,
  Runs full width below.
- **Kanban Board**: the board at full width, with the Issues panel beside it.
- **Runs & Checkpoints**: the runs table at full height.

The **Agents** button in the top bar opens a slide-over with approvals, the chief-of-staff chat and the
worker agents. Its bubble is red for failed agents and approvals waiting for you, yellow while agents run, and
green for results you have not looked at yet; opening the slide-over clears them. A finished or failed agent
also raises a toast, a desktop notification when the tab is hidden (use the bell button to allow it) and a count
in the tab title. The terminal stays in the strip at the bottom.
The commit pill copies the full hash. The sun/moon button switches between the dark and light themes
(both are defined in `web/src/tokens.css`; with no saved choice the system setting applies).

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

## Runs panel

A read-only table of training runs found under `checkpoints/` (next to the repository).
A directory counts as a run when it contains `best.pt`, `last.pt`, an `epoch<N>.pt`
snapshot, `results.json` or `results.inference.json`. Other directories (`*_splits`,
empty ones) and symlinks are skipped, and a run directory is never read through a symlink.

| Status | Meaning |
|---|---|
| Complete | `results.json` exists and has no `inference` block, or its training check says complete |
| Provisional | the training check says the run never finished; the numbers are provisional and the warnings are shown verbatim |
| Unknown | the training check could not decide (for example W&B was off); neutral, not an error |
| No results | a checkpoint exists but `results.json` does not (yet) |
| Pretrain checkpoints | SSL pretraining directories never write `results.json`; the newest epoch snapshot is shown |

Expanding a row shows its **Result** tab and, when `results.inference.json` exists, a
**Re-evaluation** tab. A re-evaluation is never shown as the fold's result. Names that do
not follow `src/training/run_naming.py` are labelled "Legacy name", and directories with a
`PIPELINE_V1` marker are labelled "Archived v1"; both are shown by default and can be
hidden with the checkbox. A corrupt or oversized JSON file gives a message on that row, not
an error for the whole panel.

**Epochs** comes from the best source available for each run:

| Shown as | Source | Means |
|---|---|---|
| `187 / 200` | the `training` block that training writes into `results.json` (runs trained after PR #134; earlier runs have none) | epochs trained out of those configured; hover shows the best epoch and whether it stopped early |
| `187 / 200` | an inference pass's training check against W&B | the same, for runs re-evaluated with `--inference-only` |
| `best 187` | the `epoch` stored inside `best.pt` | the epoch of the best validation score, **not** the epochs trained; older runs never recorded that |
| `epoch 400` | the newest `epoch<N>.pt` of a pretraining run | how far pretraining got |

The checkpoint epoch is read without loading the model: `best.pt` is an uncompressed zip, so only its
small `data.pkl` entry is read, and a restricted reader (`server/lib/pickleScan.ts`) parses the top level
of that pickle. It never imports or runs anything from the file, gives up on anything unexpected, and
results are cached until the file changes. Epochs sorts by the number shown (epochs trained, else the best epoch).

Every column header is a sort button (click again to reverse). Text columns start A to Z; metrics, epochs
and Updated start with the highest or newest. Status sorts the runs that need a human first (Provisional,
Unknown, No results, Pretrain checkpoints, Complete). Missing values always sort last, and ties fall back to the
run name. The chosen order survives the 60 s refresh and is announced to screen readers.

The table shows when each run's files last changed (the newest of `best.pt`, `last.pt`, the
newest epoch snapshot and the result files), runs start ordered by name (numbers in names compare numerically, so `-v2` comes before `-v10`),
and the table scrolls inside its panel. The name patterns are copies of
`src/training/run_naming.py`; `tests/runNamingDrift.test.ts` reads that file as text and fails if
its own examples stop classifying (it is skipped on a checkout without `src/`).

The panel only sees runs when the dashboard runs on the same filesystem as `checkpoints/`;
otherwise it says so. It never writes to `checkpoints/` and does not talk to W&B. The
browser refreshes it every 60 s and the server caches each scan for 10 s.

## Agents and approvals

The Agents drawer (top bar, right side) runs Claude Code worker agents. **New agent** takes a name, a task and an optional issue
number; the agent works alone in its own git worktree at `<agents dir>/<name>` on the branch
`phase-05-agent-<name>`, cut from `DASH_AGENT_BASE`. You can read its log, send it messages, stop it and resume
it. The main checkout is never modified: it is read-only to every agent, so queued SLURM jobs that read `src/`
from it are unaffected.

Agents use your installed `claude` and its claude.ai login (no API key), through the Claude Agent SDK. Each one
runs in safe mode (no user hooks, plugins, skills or connectors) inside a bubblewrap sandbox: the whole
filesystem is read-only except its worktree, the repository's `.git`, `~/.claude` and a private `/tmp`;
`.secrets` is hidden; the environment holds only `PATH`, `HOME`, `USER`, `LANG`, `TMPDIR` and `SHELL`.

**Approvals.** Reading, editing inside the worktree, `git status/diff/log/add/commit`, `ls`, `npm test/build`,
`pytest`, `black` and read-only `gh` commands run freely. Everything else waits on a card in the Approvals
panel that shows the exact command or file: `git push`, `gh pr create/merge/comment`, `sbatch`/`scancel`,
network tools, `rm`, and git commands that move refs or change configuration. Nothing is approved on a timeout.
The amber counter in the top bar links to the cards. Reloading the page keeps them; restarting the dashboard
stops every agent (it can be resumed).

### Chief of staff

The chat panel talks to one persistent Claude session (`DASH_CHIEF_MODEL`, Opus by default). It can read the
repository (Read, Grep, Glob) but cannot run commands or edit files. It acts only through six dashboard tools:
`list_workers` runs freely; `spawn_worker`, `message_worker`, `stop_worker`, `create_task` and `update_task` each
wait on an approval card that shows the exact arguments, so a spawned worker or a new kanban item exists only
after you approve it. Kanban changes before the first import report the usual "not imported" message. The
conversation is kept on disk (`dashboard/.state/chief/`) and resumed after a stop, a crash or a restart;
**New conversation** parks it: **Conversations** lists every conversation (title, date, turns, cost) so you can read an old one, resume it, rename it or archive it. The tool server runs inside the dashboard process (no extra port) and works under
the same safe mode and sandbox as the workers; the sandbox hides `dashboard/.state`.

Limits and caveats:

- An agent can read `~/.ssh`, `~/.config/gh` and `~/.claude` credentials inside the sandbox (an approved push needs them), so the policy
  allows no generic readers (`cat`, `curl`, `python -c`, `env`) without a card, and denies reading those paths with the file tools.
  Treat an approved command as running with your login.
- Allowed commands such as `npm test` and `python -m pytest` run code the agent wrote, and that code has the
  same network access and the same readable credentials as the agent. The approval rules gate the commands an
  agent types, not what a test script it edited does. Review the diff before trusting a worktree.
- The sandbox hides `dashboard/.state` (the access token) and `.secrets`, and keeps `~/.claude` settings, hooks,
  skills and plugins read-only. The repository's `.git` stays writable so commits work; refs of other branches
  are made read-only at session start, but a worker could still create a new loose ref, so treat the
  repository metadata as the agent's to touch.
- The SDK's own sandbox stalls on this node's NFS home, which is why bubblewrap is used directly.
- `DASH_AGENT_DRIVER=fake` plays scripted sessions for tests; the browser tests refuse to start with anything else.
- Pinned versions that were tested together: `@anthropic-ai/claude-agent-sdk` 0.3.304 with `claude` 2.1.292.

## Terminal

The drawer at the bottom opens a terminal running your normal login shell, like ssh. It runs inside a tmux
session (`DASH_TERMINAL_SESSION`, default `hitfinder-dash`), so closing the drawer, reloading the page or
restarting the dashboard leaves the shell, and anything running in it such as `claude`, going; opening the drawer
again re-attaches. From an ssh session you can join the same shell with `tmux attach -t hitfinder-dash`.

How it works: the browser opens a stream (`POST /api/terminal/open`, then a streaming `GET`), sends keystrokes and
resizes as `POST`s, and a small Python helper (`server/terminal/ptyBridge.py`, standard library only) owns the PTY.
There is no WebSocket server and no native add-on. Up to `DASH_TERMINAL_MAX` (default 2) streams at once.
Keyboard: xterm keeps Tab for the shell, so press F6 to move focus to the Close button.

**This is a full shell as you, protected only by the dashboard token.** It is not sandboxed. The server stays on
`127.0.0.1`, the token and cookie are required, the stream refuses requests that come from another origin, and
agents cannot reach it (their sandbox hides `dashboard/.state`). Anyone who gets the token has your account on
this node, so treat the tokenised URL like a password, and set `DASH_TERMINAL=off` to switch the terminal off.
The shell inherits only `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, the locale variables, `TZ` and `TMPDIR`
(plus `TERM`); your login profile sets up the rest, as it does over ssh, so no credential variables are passed on.
The server also answers only requests addressed to `localhost`, `127.0.0.1` or `[::1]` (any port, so an ssh tunnel
on another local port works), which blocks DNS-rebinding pages.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DASH_PORT` | `4317` | Port on `127.0.0.1` |
| `DASH_REPO_ROOT` | parent of `dashboard/` | Repository the panels describe |
| `DASH_STATE_DIR` | `dashboard/.state` | Token, junit report, sync state and lock |
| `DASH_KANBAN` | `<repo>/phase-05-kanban.md` | Kanban markdown file |
| `DASH_CHECKPOINTS` | `<repo>/checkpoints` | Directory scanned by the Runs panel |
| `DASH_GH_BIN` | `gh` | GitHub CLI binary |
| `DASH_PYTHON` | `python` | Interpreter used for the local test run |
| `DASH_AGENTS_DIR` | `<repo>-agents` (next to the repo) | Where agent worktrees are created |
| `DASH_AGENT_BASE` | `origin/phase-05` | Branch or ref new agent branches start from |
| `DASH_AGENT_MAX` | `3` | Agents allowed at once, 1 to 8 |
| `DASH_AGENT_MAX_TURNS` | `60` | Turn limit per agent session, 1 to 500 |
| `DASH_WORKER_MODEL` | `sonnet` | Model for worker agents |
| `DASH_CHIEF_MODEL` | `opus` | Model for the chief of staff |
| `DASH_CLAUDE_BIN` | `claude` | Claude Code binary the agents drive |
| `DASH_AGENT_DRIVER` | `sdk` | `sdk` (real) or `fake` (scripted, for tests) |
| `DASH_TERMINAL` | `on` | `off` switches the terminal off |
| `DASH_TERMINAL_SESSION` | `hitfinder-dash` | tmux session the terminal attaches to |
| `DASH_TERMINAL_MAX` | `2` | Terminal streams at once, 1 to 4 |
| `DASH_TERMINAL_PYTHON` | `python3` | Interpreter for the PTY helper (3.6 or newer) |

An empty or whitespace-only value counts as unset and uses the default, so `DASH_CHECKPOINTS=$D npm start`
with `$D` unset in that shell does not fall back to the current directory. `DASH_PORT` is the exception: a
blank port is an error. Check what a running server uses with `tr '\0' '\n' < /proc/<pid>/environ | grep ^DASH_`.

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
parser, `lib/kanbanSync.ts` the sync, `lib/runScan.ts` the Runs scanner), `web/src/` (React panels and components, plain CSS: global
tokens in `tokens.css`, shared base in `app.css`, and one CSS file beside each component; Geist fonts via fontsource), `shared/types.ts` (API types), `tests/`, `e2e/`.
All subprocesses (`gh`, `git`, `pytest`) are run with argument arrays, never
through a shell.
