#!/usr/bin/env python3
"""PreToolUse guard for Bash commands. A seatbelt, not a sandbox.

Reads the hook JSON on stdin and prints a PreToolUse permission decision.

deny  - force-push to main or phase-NN
      - `git add -f` of docs/superpowers/, PLANNING.md, MEMORY.md, the kanban file
      - `git add` of anything under .secrets/
      - a reader command (cat, grep, cp, ...) on a file inside .secrets/
ask   - recursive rm touching checkpoints/, logs/, data/ or the frame cache
      - git reset --hard, git clean (unless dry-run)
      - a branch switch in the main checkout
      - scancel on anything other than one plain job id
      - `gh pr create` without a `ready` /sfx-evaluate code report for the
        current branch and commit
      - any other command naming a file inside .secrets/ (source, ls, stat,
        test, chmod, ln and mkdir are allowed)
allow - everything else (no output)

It matches command text only. A destructive command inside a script it does
not read will pass; so will `find -delete` and `rsync --delete`.

Keep this file Python 3.6 compatible and stdlib only: hooks may run under
/usr/bin/python3. If the script itself fails, the decision is "ask".
"""

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

PREFILTER_RE = re.compile(r"\b(rm|git|scancel|gh)\b|\.secrets")
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
SECRETS_READ_CMDS = (
    "cat", "tac", "head", "tail", "less", "more", "nl", "wc", "grep", "egrep",
    "fgrep", "rg", "ag", "ack", "awk", "gawk", "sed", "cut", "sort", "uniq",
    "strings", "xxd", "od", "hexdump", "base64", "cp", "mv", "scp", "rsync",
    "tar", "zip", "gzip", "diff", "cmp", "tee", "dd", "vi", "vim", "nano",
    "emacs", "python", "python3", "perl", "curl", "wget", "md5sum", "sha256sum",
)  # fmt: skip
WRAPPERS = ("sudo", "command", "time", "nohup", "exec", "then", "do", "else", "!")
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


def _argv(segment):
    try:
        toks = shlex.split(segment)
    except ValueError:
        toks = segment.split()
    while toks:
        toks[0] = toks[0].lstrip("({")
        if not toks[0] or toks[0] in WRAPPERS or re.match(r"^[A-Za-z_]\w*=", toks[0]):
            toks = toks[1:]
        else:
            break
    return toks


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


def _check_git(argv, cwd, ctx):
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        if argv[i] == "-C" and i + 1 < len(argv):
            cwd = os.path.normpath(os.path.join(cwd, argv[i + 1]))
        i += 2 if argv[i] in ("-C", "-c") else 1
    if i >= len(argv):
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


def _check_secrets(argv):
    hits = [t for t in argv if ".secrets/" in t and not t.endswith(".secrets/")]
    if not hits:
        return ALLOW, ""
    cmd = os.path.basename(argv[0])
    if cmd in SECRETS_OK_CMDS:
        return ALLOW, ""
    if cmd == "git" and any(a in SECRETS_OK_GIT_SUBCMDS for a in argv[1:]):
        return ALLOW, ""
    if cmd in SECRETS_READ_CMDS:
        return (
            DENY,
            "reading files in .secrets/ is blocked (use `source .secrets/wandb.env`)",
        )
    return ASK, "command names a file inside .secrets/"


def _check_segment(segment, cwd, ctx):
    argv = _argv(segment)
    if not argv:
        return ALLOW, ""
    cmd = os.path.basename(argv[0])
    results = [_check_secrets(argv)]
    if cmd in SHELLS and "-c" in argv[:-1]:
        results.append(decide(argv[argv.index("-c") + 1], cwd, ctx))
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


def decide(command, cwd, ctx):
    """Return (action, reason) for a Bash command line; the strictest segment wins."""
    worst = (ALLOW, "")
    for segment in _segments(command):
        argv = _argv(segment)
        if argv and argv[0] == "cd" and len(argv) > 1:
            cwd = os.path.normpath(os.path.join(cwd, os.path.expanduser(argv[1])))
            continue
        result = _check_segment(segment, cwd, ctx)
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
        if not PREFILTER_RE.search(command):
            return 0
        action, reason = decide(
            command, payload.get("cwd") or os.getcwd(), GitContext()
        )
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
