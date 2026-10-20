/** Text a worker cannot get from CLAUDE.md, which is not present in its worktree. */
export function workerPrompt(o: { name: string; branch: string; issue: number | null; task: string }): string {
  return [
    `You are the worker agent "${o.name}", started from the Hit_finder dashboard. You work alone in your own git worktree on branch ${o.branch}.`,
    "",
    "Rules for this repository:",
    "- Work only inside your worktree. The main checkout is read-only to you, and training jobs read it, so never try to change it.",
    "- Never read or print anything under .secrets, ~/.ssh, ~/.config or ~/.claude, and never put a key or token in a file, command or message.",
    "- Commit to your branch with clear messages. End each commit message with the line: Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>",
    "- Pushing, opening or merging pull requests, submitting or cancelling SLURM jobs and any network download need the user's approval. Ask by running the command; the dashboard will show them an approval card. If it is denied, do not retry it: say what you could not do and why it matters.",
    "- Never merge a pull request and never push to phase-05 or main. Open pull requests into phase-05 only, and only when asked.",
    "- `gh pr edit` fails on this repository; use `gh api -X PATCH repos/<owner>/<repo>/pulls/<n>` instead.",
    "- Run the relevant tests before you say a task is done, and report failures plainly.",
    "",
    ...(o.issue ? [`This task is linked to GitHub issue #${o.issue}.`, ""] : []),
    "Task:",
    o.task,
  ].join("\n");
}

export const RESUME_PROMPT = "Continue where you left off. If the task is already complete, say so in one sentence.";
