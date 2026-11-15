#!/usr/bin/env python3
"""PreToolUse guard for Bash commands. A seatbelt, not a sandbox.

Reads the hook JSON on stdin and prints a PreToolUse permission decision.

deny  - force-push to main or phase-NN
      - `git add -f` of docs/superpowers/, PLANNING.md, MEMORY.md, the kanban file
      - `git add` of anything under .secrets/
      - a reader command (cat, grep, cp, git grep, ...) on .secrets or a file
        inside it, or on /proc/*/environ, however the path is written (quotes,
        trailing slash, dot-glob such as .secret?, {a,b} group), or after a
        cd/pushd into .secrets; also inside `bash -c` and `eval`
      - echo/printf expanding $WANDB_API_KEY; printenv/declare naming it
      - when the command line sources a .secrets file: env, printenv,
        export/declare/typeset with no names, bare set, set -x/-v, bash -x
        (they would print the key)
ask   - any other command using WANDB_API_KEY (test, [, unset are fine), or
        reading the environment on a line that sources .secrets/
      - a VAR=value assignment naming .secrets
      - recursive rm touching checkpoints/, logs/, data/ or the frame cache
      - git reset --hard, git clean (unless dry-run)
      - a branch switch in the main checkout
      - scancel on anything other than one plain job id
      - `gh pr create` without a `ready` /sfx-evaluate code report for the
        current branch and commit
      - any other command naming a file inside .secrets/ (source, ls, stat,
        test, chmod, ln and mkdir are allowed); a redirect such as
        `tr a a < .secrets/x` lands here
allow - everything else (no output). grep/rg/git grep patterns and git -m,
        --grep, gh --title/--body text are not treated as file names.

It matches command text only. A destructive command inside a script it does
not read will pass; so will `find -delete` and `rsync --delete`. Reads of
.secrets/ also pass when nothing on the line names it: `grep -r KEY .`, a
symlink to it, `shopt -s dotglob`, ANSI-C quoting such as $'\\x2esecrets', a
program that prints os.environ without sourcing on the same line. Heredoc
bodies are checked line by line, as commands.

Keep this file Python 3.6 compatible and stdlib only: hooks may run under
/usr/bin/python3. If the script itself fails, the decision is "ask".
"""

import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys

ALLOW = "allow"
ASK = "ask"
DENY = "deny"
_RANK = {ALLOW: 0, ASK: 1, DENY: 2}

SECRETS_DIR = ".secrets"
SECRETS_NAME_RE = re.compile(r"(^|[^\w.-])\.secrets($|[^\w.-])")
PROC_ENVIRON_RE = re.compile(r"/proc/.*environ")
BRACE_RE = re.compile(r"\{([^{}]*,[^{}]*)\}")
MAX_BRACE_WORDS = 64
ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")
KEY_NAME_RE = re.compile(r"\bWANDB_API_KEY\b")
KEY_EXPAND_RE = re.compile(r"\$\{?WANDB_API_KEY\b")
SOURCE_CMDS = ("source", ".")
CD_CMDS = ("cd", "pushd")
ECHO_CMDS = ("echo", "printf")
KEY_NAME_PRINT_CMDS = ("printenv", "declare", "typeset")
KEY_SAFE_CMDS = ("test", "[", "[[", ":", "unset")
VAR_DUMP_CMDS = ("declare", "typeset", "export")
TRACE_FLAGS = "xv"
TRACE_OPTIONS = {"xtrace": "x", "verbose": "v"}
GREP_CMDS = ("grep", "egrep", "fgrep", "rg", "ag", "ack")
# grep/rg options whose value is the next token (so it is not the pattern).
GREP_VALUE_OPTS = (
    "-A", "-B", "-C", "-m", "-d", "-D", "-f", "-g", "-t", "-T", "-M",
    "--file", "--glob", "--type", "--max-count", "--context",
    "--after-context", "--before-context",
)  # fmt: skip
GIT_TEXT_OPTS = ("-m", "--message", "--grep")
GH_TEXT_OPTS = ("-t", "--title", "-b", "--body")
PROTECTED_BRANCH_RE = re.compile(r"^(main|phase-\d{2})$")
PROTECTED_DIR_NAMES = ("checkpoints", "logs", "data")
# Basenames of cache.root and cache.nvme_root in configs/base.yaml.
# tests/test_guard_hook.py fails if the two drift apart.
CACHE_DIR_NAMES = ("Hit_finder_cache", "sfx_frame_cache")
PRIVATE_DOC_RE = re.compile(
    r"(^|/)(docs/superpowers(/.*)?|PLANNING\.md|MEMORY\.md|phase-\d{2}-kanban\.md)$"
)
FORCE_ADD_PARENTS = (".", "docs", ":/", "*")
SECRETS_OK_CMDS = (
    "source",
    ".",
    "ls",
    "stat",
    "test",
    "[",
    "[[",
    "chmod",
    "ln",
    "mkdir",
)
SECRETS_OK_GIT_SUBCMDS = ("check-ignore", "ls-files", "status")
SECRETS_READ_GIT_SUBCMDS = ("grep", "show", "cat-file")
SECRETS_READ_CMDS = (
    "cat", "tac", "head", "tail", "less", "more", "nl", "wc", "grep", "egrep",
    "fgrep", "rg", "ag", "ack", "awk", "gawk", "sed", "cut", "sort", "uniq",
    "strings", "xxd", "od", "hexdump", "base64", "cp", "mv", "scp", "rsync",
    "tar", "zip", "gzip", "diff", "cmp", "tee", "dd", "vi", "vim", "nano",
    "emacs", "python", "python3", "perl", "curl", "wget", "md5sum", "sha256sum",
)  # fmt: skip
WRAPPERS = (
    "sudo", "command", "builtin", "time", "nohup", "exec", "then", "do", "else", "!",
)  # fmt: skip
SHELLS = ("bash", "sh", "zsh")
SCANCEL_HARMLESS_FLAGS = ("-v", "--verbose", "-Q", "--quiet")
JOB_ID_RE = re.compile(r"^\d+$")
GLOB_CHARS = "*?["
REPORT_HEADER_LINES = 20
REPORT_MARKER = "-evaluate-code-"
MIN_COMMIT_CHARS = 7
GIT_TIMEOUT_S = 5


def _segments(command):
    """Split a command line into simple commands, including $(...) and `...` bodies."""
    inner = re.findall(r"\$\(([^()]*)\)", command) + re.findall(r"`([^`]*)`", command)
    parts = re.split(r"&&|\|\||[;|\n&]", command)
    return [p for p in parts + inner if p.strip()]


def _split(segment):
    """Return (leading VAR=value assignments, argv) for one simple command.

    Drops wrappers such as sudo, and the ( { ) } of subshells and groups.
    """
    try:
        toks = shlex.split(segment)
    except ValueError:
        toks = segment.split()
    assigns = []
    while toks:
        toks[0] = toks[0].lstrip("({")
        if ASSIGN_RE.match(toks[0]):
            assigns.append(toks.pop(0))
        elif not toks[0] or toks[0] in WRAPPERS:
            toks.pop(0)
        else:
            break
    if toks:
        toks[-1] = toks[-1].rstrip(")}")
        if not toks[-1]:
            toks.pop()
    return assigns, toks


def _argv(segment):
    return _split(segment)[1]


def _has_short_flag(args, letters):
    """True if a short-option cluster such as -rf contains one of `letters`."""
    for a in args:
        if a == "--":
            return False
        if re.match(r"^-[A-Za-z]+$", a) and any(c in a for c in letters):
            return True
    return False


def _positionals(args):
    out, only_paths = [], False
    for a in args:
        if only_paths or a == "-" or not a.startswith("-"):
            out.append(a)
        elif a == "--":
            only_paths = True
    return out


def _touches_protected(token, cwd, roots):
    parts = [p for p in token.split("/") if p]
    if any(p in CACHE_DIR_NAMES for p in parts):
        return True
    if "$" in token or "`" in token:
        tail = parts[1:] if token.startswith("/") else parts
        return any(p in ("checkpoints", "logs") for p in parts) or "data" in tail
    path = os.path.normpath(os.path.join(cwd, os.path.expanduser(token)))
    comps = path.split(os.sep)
    for i, comp in enumerate(comps):
        if any(c in comp for c in GLOB_CHARS):
            path = os.sep.join(comps[:i]) or os.sep
            break
    for root in roots:
        for name in PROTECTED_DIR_NAMES:
            prot = os.path.join(root, name)
            inside = path == prot or path.startswith(prot + os.sep)
            ancestor = prot.startswith(path.rstrip(os.sep) + os.sep)
            if inside or ancestor:
                return True
    return False


def _check_rm(args, cwd, ctx):
    if not (_has_short_flag(args, "rR") or "--recursive" in args):
        return ALLOW, ""
    roots = ctx.checkout_roots(cwd)
    for target in _positionals(args):
        if _touches_protected(target, cwd, roots):
            return ASK, "recursive rm touches a protected directory: %s" % target
    return ALLOW, ""


def _check_git_push(args, cwd, ctx):
    refs = _positionals(args)[1:]
    forced = (
        _has_short_flag(args, "f")
        or any(a.startswith("--force") for a in args)
        or any(r.startswith("+") for r in refs)
    )
    if not forced:
        return ALLOW, ""
    if "--all" in args or "--mirror" in args:
        return DENY, "force-push of all branches is blocked"
    targets = []
    for ref in refs:
        dst = ref.lstrip("+").split(":")[-1]
        if dst.startswith("refs/heads/"):
            dst = dst[len("refs/heads/") :]
        targets.append(ctx.current_branch(cwd) if dst == "HEAD" else dst)
    if not targets:
        targets = [ctx.current_branch(cwd)]
    for target in targets:
        if PROTECTED_BRANCH_RE.match(target or ""):
            return DENY, "force-push to %s is blocked" % target
    return ALLOW, ""


def _check_git_add(args):
    paths = _positionals(args)
    if any(".secrets" in p.split("/") for p in paths):
        return DENY, "git add of .secrets/ is blocked"
    if not (_has_short_flag(args, "f") or "--force" in args):
        return ALLOW, ""
    for p in paths:
        norm = os.path.normpath(p) if p not in FORCE_ADD_PARENTS else p
        if PRIVATE_DOC_RE.search(norm) or norm in FORCE_ADD_PARENTS:
            return DENY, "git add -f would stage a local-only document: %s" % p
    if not paths:
        return DENY, "git add -f with no path could stage local-only documents"
    return ALLOW, ""


def _check_git_checkout(sub, args, cwd, ctx):
    if not ctx.is_main_checkout(cwd):
        return ALLOW, ""
    reason = "branch switch in the main checkout (SLURM jobs read it live)"
    if sub == "switch":
        return (ALLOW, "") if "-h" in args or "--help" in args else (ASK, reason)
    if "--" in args:
        return ALLOW, ""
    if any(a in ("-b", "-B", "--detach", "--orphan", "-t", "--track") for a in args):
        return ASK, reason
    targets = _positionals(args)
    if len(targets) != 1 or ctx.path_exists(cwd, targets[0]):
        return ALLOW, ""  # no target, or a file restore
    return ASK, reason


def _git_subcommand(argv, cwd):
    """Return (index of the subcommand in argv or None, cwd after any -C)."""
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        if argv[i] == "-C" and i + 1 < len(argv):
            cwd = os.path.normpath(os.path.join(cwd, argv[i + 1]))
        i += 2 if argv[i] in ("-C", "-c") else 1
    return (i if i < len(argv) else None), cwd


def _check_git(argv, cwd, ctx):
    i, cwd = _git_subcommand(argv, cwd)
    if i is None:
        return ALLOW, ""
    sub, args = argv[i], argv[i + 1 :]
    if sub == "push":
        return _check_git_push(args, cwd, ctx)
    if sub == "add":
        return _check_git_add(args)
    if sub == "reset" and "--hard" in args:
        return ASK, "git reset --hard discards uncommitted work"
    if sub == "clean" and not (_has_short_flag(args, "n") or "--dry-run" in args):
        return (
            ASK,
            "git clean deletes untracked files (local-only docs live untracked here)",
        )
    if sub in ("checkout", "switch"):
        return _check_git_checkout(sub, args, cwd, ctx)
    return ALLOW, ""


def _check_scancel(args):
    rest = [a for a in args if a not in SCANCEL_HARMLESS_FLAGS]
    if len(rest) == 1 and JOB_ID_RE.match(rest[0]):
        return ALLOW, ""
    return ASK, "scancel on more than one job, or by user/name/state"


def _brace_words(token):
    """Expand simple {a,b} groups the way the shell would (capped)."""
    todo, out = [token], []
    while todo and len(out) < MAX_BRACE_WORDS:
        word = todo.pop()
        m = BRACE_RE.search(word)
        if not m:
            out.append(word)
            continue
        for alt in m.group(1).split(","):
            todo.append(word[: m.start()] + alt + word[m.end() :])
    return out + todo


def _names_secrets(token):
    """True if `token` names .secrets, or a dot-glob or {a,b} group can expand to it."""
    for word in _brace_words(token):
        candidates = [word] + ([word.split("=", 1)[1]] if "=" in word else [])
        for cand in candidates:
            if SECRETS_NAME_RE.search(cand):
                return True
            for comp in cand.split("/"):
                globbed = comp.startswith(".") and any(c in comp for c in GLOB_CHARS)
                if globbed and fnmatch.fnmatchcase(SECRETS_DIR, comp):
                    return True
    return False


def _sources_secrets(argv, in_secrets):
    if not argv or argv[0] not in SOURCE_CMDS:
        return False
    return in_secrets or any(_names_secrets(t) for t in argv[1:])


def _option_values(argv, start, opts):
    """Indices of the values of `opts` (`-m msg`, `--body=text`, `-am msg`)."""
    out = set()
    for i in range(start, len(argv)):
        a = argv[i]
        if a in opts or ("-m" in opts and re.match(r"^-[A-Za-z]+m$", a)):
            out.add(i + 1)
        elif "=" in a and a.split("=", 1)[0] in opts:
            out.add(i)
    return out


def _grep_pattern(argv, start):
    """Index of the search pattern(s) of a grep-like command starting at `start`."""
    out, i = set(), start
    while i < len(argv):
        a = argv[i]
        if a == "--":
            if not out and i + 1 < len(argv):
                out.add(i + 1)
            return out
        if a in ("-e", "--regexp") or re.match(r"^-[A-Za-z]+e$", a):
            out.add(i + 1)
            i += 2
        elif a.startswith("--regexp="):
            out.add(i)
            i += 1
        elif a in GREP_VALUE_OPTS:
            i += 2
        elif a.startswith("-") and a != "-":
            i += 1
        else:
            if not out:
                out.add(i)
            return out
    return out


def _text_args(argv, cwd):
    """Indices of argv that hold search patterns or message text, not file names."""
    cmd = os.path.basename(argv[0])
    if cmd in GREP_CMDS:
        return _grep_pattern(argv, 1)
    if cmd == "git":
        sub = _git_subcommand(argv, cwd)[0]
        if sub is None:
            return set()
        if argv[sub] == "grep":
            return _grep_pattern(argv, sub + 1)
        return _option_values(argv, sub + 1, GIT_TEXT_OPTS)
    if cmd == "gh":
        return _option_values(argv, 1, GH_TEXT_OPTS)
    return set()


def _env_dumps(args):
    """True if `env args` prints the environment (no command, or options first)."""
    i = 0
    while i < len(args) and ASSIGN_RE.match(args[i]):
        i += 1
    return i >= len(args) or args[i].startswith("-")


def _shell_command(args):
    """Return (the -c command string or None, option letters) for bash/sh args."""
    flags, i = "", 0
    while i < len(args):
        a = args[i]
        if a in ("-o", "+o") and i + 1 < len(args):
            flags += TRACE_OPTIONS.get(args[i + 1], "") if a == "-o" else ""
            i += 2
        elif re.match(r"^-[A-Za-z]+$", a):
            flags += a[1:]
            i += 1
        else:
            break
    if "c" not in flags or i >= len(args):
        return None, flags
    return args[i], flags


def _check_key(argv, text, sourced):
    """Deny commands that would print WANDB_API_KEY; ask on other uses of it."""
    cmd, args = os.path.basename(argv[0]), argv[1:]
    reason = "this would print WANDB_API_KEY"
    if cmd in ECHO_CMDS and any(KEY_EXPAND_RE.search(a) for a in args):
        return DENY, reason
    if cmd in KEY_NAME_PRINT_CMDS and any(KEY_NAME_RE.search(a) for a in args):
        return DENY, reason
    if sourced:
        dump = reason + " (%s on a line that sources .secrets/)"
        if cmd == "printenv" and not _positionals(args):
            return DENY, dump % "environment dump"
        if cmd == "env" and _env_dumps(args):
            return DENY, dump % "environment dump"
        if cmd in VAR_DUMP_CMDS and not _positionals(args):
            return DENY, dump % "variable dump"
        if cmd == "set" and not args:
            return DENY, dump % "variable dump"
        traced = _has_short_flag(args, TRACE_FLAGS)
        if cmd == "set" and (traced or any(a in TRACE_OPTIONS for a in args)):
            return DENY, dump % "shell tracing"
    used = [
        a
        for i, a in enumerate(argv)
        if KEY_NAME_RE.search(a) and (i not in text or KEY_EXPAND_RE.search(a))
    ]
    if used and cmd not in KEY_SAFE_CMDS + ECHO_CMDS:
        return ASK, "command uses WANDB_API_KEY"
    if sourced and any("environ" in a.lower() for a in argv):
        return ASK, "command reads the environment on a line that sources .secrets/"
    return ALLOW, ""


def _check_secrets(argv, text, in_secrets=False):
    hits = [
        t
        for i, t in enumerate(argv)
        if i not in text and (_names_secrets(t) or PROC_ENVIRON_RE.search(t))
    ]
    if not hits and not in_secrets:
        return ALLOW, ""
    cmd = os.path.basename(argv[0])
    if cmd in SECRETS_OK_CMDS:
        return ALLOW, ""
    sub = None
    if cmd == "git":
        i = _git_subcommand(argv, "")[0]
        sub = argv[i] if i is not None else None
    if sub in SECRETS_OK_GIT_SUBCMDS:
        return ALLOW, ""
    if cmd in SECRETS_READ_CMDS or sub in SECRETS_READ_GIT_SUBCMDS:
        return (
            DENY,
            "reading .secrets/ or a process environment is blocked"
            " (use `source .secrets/wandb.env`)",
        )
    return ASK, "command names a file inside .secrets/"


def _check_segment(segment, cwd, ctx, in_secrets=False, sourced=False):
    assigns, argv = _split(segment)
    results = [(ALLOW, "")]
    if any(_names_secrets(a.split("=", 1)[1]) for a in assigns):
        results.append((ASK, "a variable is set to a path in .secrets/"))
    if not argv:
        return max(results, key=lambda r: _RANK[r[0]])
    cmd = os.path.basename(argv[0])
    text = _text_args(argv, cwd)
    results.append(_check_secrets(argv, text, in_secrets))
    results.append(_check_key(argv, text, sourced))
    body, flags = _shell_command(argv[1:]) if cmd in SHELLS else (None, "")
    if body is not None:
        results.append(decide(body, cwd, ctx, sourced))
        traced = any(f in flags for f in TRACE_FLAGS)
        if traced and _plan(body, cwd, sourced)[1]:
            results.append(
                (DENY, "shell tracing would print the sourced .secrets/ file")
            )
    elif cmd == "eval":
        results.append(decide(" ".join(argv[1:]), cwd, ctx, sourced))
    elif cmd == "xargs":
        if "rm" in argv and _has_short_flag(argv[argv.index("rm") :], "rR"):
            results.append((ASK, "xargs rm -r: targets come from stdin"))
        if "scancel" in argv:
            results.append((ASK, "xargs scancel: more than one job"))
    elif cmd == "rm":
        results.append(_check_rm(argv[1:], cwd, ctx))
    elif cmd == "git":
        results.append(_check_git(argv, cwd, ctx))
    elif cmd == "scancel":
        results.append(_check_scancel(argv[1:]))
    elif cmd == "gh" and argv[1:3] == ["pr", "create"]:
        status = ctx.eval_report_status(cwd)
        if status != "ready":
            results.append(
                (
                    ASK,
                    "no ready /sfx-evaluate code report for this branch and commit (status: %s)"
                    % status,
                )
            )
    return max(results, key=lambda r: _RANK[r[0]])


def _cd_target(argv, cwd):
    targets = [a for a in _positionals(argv[1:]) if a != "-" and not a.startswith("+")]
    if not targets:
        return os.path.expanduser("~") if argv == ["cd"] else cwd
    return os.path.normpath(os.path.join(cwd, os.path.expanduser(targets[0])))


def _plan(command, cwd, sourced=False):
    """Return ([(segment, cwd, cwd is inside .secrets)], whether .secrets/ is sourced).

    cd/pushd segments only move cwd; they are not checked themselves.
    """
    steps = []
    for segment in _segments(command):
        argv = _argv(segment)
        if argv and argv[0] in CD_CMDS:
            cwd = _cd_target(argv, cwd)
            continue
        in_secrets = _names_secrets(cwd)
        sourced = sourced or _sources_secrets(argv, in_secrets)
        steps.append((segment, cwd, in_secrets))
    return steps, sourced


def decide(command, cwd, ctx, sourced=False):
    """Return (action, reason) for a Bash command line; the strictest segment wins.

    `sourced` is True when an enclosing command line already sourced .secrets/.
    """
    steps, sourced = _plan(command, cwd, sourced)
    worst = (ALLOW, "")
    for segment, seg_cwd, in_secrets in steps:
        result = _check_segment(segment, seg_cwd, ctx, in_secrets, sourced)
        if _RANK[result[0]] > _RANK[worst[0]]:
            worst = result
    return worst


def _read_header(path):
    header = {}
    with open(path, "r") as fh:
        for _ in range(REPORT_HEADER_LINES):
            line = fh.readline()
            if ":" in line:
                key, _, value = line.partition(":")
                header[key.strip().lower()] = value.strip()
    return header


def report_status(reports_dir, branch, commit):
    """Verdict of the newest evaluate-code report for (branch, commit), or 'missing'."""
    if not branch or not commit or not os.path.isdir(reports_dir):
        return "missing"
    matches = []
    for name in os.listdir(reports_dir):
        if REPORT_MARKER not in name or not name.endswith(".md"):
            continue
        path = os.path.join(reports_dir, name)
        header = _read_header(path)
        sha = header.get("commit", "")
        if header.get("workflow") != "evaluate-code" or header.get("branch") != branch:
            continue
        if len(sha) < MIN_COMMIT_CHARS or not commit.startswith(sha):
            continue
        matches.append((os.path.getmtime(path), header.get("verdict", "") or "missing"))
    return max(matches)[1] if matches else "missing"


class GitContext(object):
    """Answers the questions decide() asks about the real checkout."""

    def _git(self, cwd, *args):
        try:
            proc = subprocess.run(
                ["git", "-C", cwd] + list(args),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
                timeout=GIT_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""

    def main_root(self, cwd):
        first = self._git(cwd, "worktree", "list", "--porcelain").split("\n", 1)[0]
        return first[len("worktree ") :] if first.startswith("worktree ") else ""

    def checkout_roots(self, cwd):
        roots = [self._git(cwd, "rev-parse", "--show-toplevel"), self.main_root(cwd)]
        return [r for r in roots if r]

    def is_main_checkout(self, cwd):
        top, main = self._git(cwd, "rev-parse", "--show-toplevel"), self.main_root(cwd)
        return bool(top) and os.path.realpath(top) == os.path.realpath(main)

    def current_branch(self, cwd):
        return self._git(cwd, "branch", "--show-current")

    def path_exists(self, cwd, name):
        return os.path.exists(os.path.join(cwd, name))

    def eval_report_status(self, cwd):
        main = self.main_root(cwd)
        if not main:
            return "missing"
        return report_status(
            os.path.join(main, "reports", "workflows"),
            self.current_branch(cwd),
            self._git(cwd, "rev-parse", "HEAD"),
        )


def main():
    try:
        payload = json.load(sys.stdin)
        command = (payload.get("tool_input") or {}).get("command") or ""
        cwd = payload.get("cwd") or os.getcwd()
        action, reason = decide(command, cwd, GitContext())
    except Exception as exc:  # a broken guard must ask, never silently allow
        action = ASK
        reason = "guard hook error (%s: %s); check this command yourself" % (
            type(exc).__name__,
            exc,
        )
    if action != ALLOW:
        decision = {
            "hookEventName": "PreToolUse",
            "permissionDecision": action,
            "permissionDecisionReason": "guard: " + reason,
        }
        print(json.dumps({"hookSpecificOutput": decision}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
