import fs from "node:fs";
import path from "node:path";
import { test, expect, type Page } from "@playwright/test";
import { startDashboard, type Dashboard } from "./harness";

/** Starts one dashboard (own port, own temp directory, fake gh and pytest) for the tests of a describe block. */
function withDashboard(opts: Parameters<typeof startDashboard>[0] = {}): () => Dashboard {
  let dash: Dashboard | null = null;
  test.beforeAll(async () => {
    dash = await startDashboard(opts);
  });
  test.afterAll(async () => {
    await dash?.stop();
    dash = null;
  });
  return () => {
    if (!dash) throw new Error("the dashboard is not running");
    return dash;
  };
}

/** Opens the tokenised URL, as a user would from the address the server prints. */
async function open(page: Page, dash: Dashboard): Promise<void> {
  await page.goto(dash.tokenUrl);
  await expect(page.getByRole("heading", { level: 1, name: "Hit_finder" })).toBeVisible();
}

const panel = (page: Page, title: string) => page.getByRole("region", { name: title, exact: true });
const column = (page: Page, name: string) => page.getByRole("region", { name: new RegExp(`^${name}`) });
const createCalls = (dash: Dashboard) => dash.ghCalls().filter((c) => c.args[0] === "issue" && c.args[1] === "create");

test.describe("access", () => {
  const dashboard = withDashboard();

  test("an unauthenticated visit is refused", async ({ page, request }) => {
    const dash = dashboard();
    const res = await page.goto(dash.baseUrl + "/");
    expect(res?.status()).toBe(401);
    await expect(page.locator("body")).toContainText("Unauthorized");
    expect((await request.get(`${dash.baseUrl}/api/prs`)).status()).toBe(401);
    expect((await request.get(`${dash.baseUrl}/?token=wrong`)).status()).toBe(401);
  });

  test("the tokenised URL sets the cookie, lands on the app and removes the token from the address bar", async ({ page, context }) => {
    const dash = dashboard();
    await page.goto(dash.tokenUrl);
    await expect(page.getByRole("heading", { level: 1, name: "Hit_finder" })).toBeVisible();
    expect(page.url()).toBe(dash.baseUrl + "/");
    expect(page.url()).not.toContain(dash.token);
    const cookie = (await context.cookies()).find((c) => c.name === "dash_token");
    expect(cookie?.httpOnly).toBe(true);
    expect(cookie?.sameSite).toBe("Strict");
    // the cookie alone now opens the app, and the API
    await page.goto(dash.baseUrl + "/");
    await expect(page.getByRole("heading", { level: 1, name: "Hit_finder" })).toBeVisible();
    expect((await page.request.get(`${dash.baseUrl}/api/health`)).status()).toBe(200);
  });
});

test.describe("panels with data", () => {
  const dashboard = withDashboard();

  test("the top bar shows the branch of the repository", async ({ page }) => {
    await open(page, dashboard());
    await expect(page.getByLabel("Current branch")).toHaveText("e2e-branch");
  });

  test("latest pull request shows the summary bullets and the open PR's check state", async ({ page }) => {
    await open(page, dashboard());
    const pr = panel(page, "Latest pull request");
    await expect(pr.getByRole("link", { name: "Add the status dashboard" })).toBeVisible();
    const bullets = pr.getByRole("list", { name: "What it did" }).getByRole("listitem");
    await expect(bullets).toHaveText(["Shows the latest pull request", "Syncs the kanban file with GitHub issues"]);
    const openPrs = pr.getByRole("list", { name: "Open pull requests" });
    await expect(openPrs.getByRole("link", { name: "Tune the hit finder" })).toBeVisible();
    await expect(openPrs).toContainText("Checks failing (1 of 3)");
    await expect(pr.getByRole("list", { name: "Earlier pull requests" })).toContainText("Add frame cache");
  });

  test("issues lists the ordinary open issue and the CI panel lists the runs", async ({ page }) => {
    await open(page, dashboard());
    await expect(panel(page, "Issues").getByRole("list", { name: "Open issues" })).toContainText("Eiger4M geometry looks off");
    await expect(panel(page, "Tests").getByText("GitHub Actions")).toBeVisible();
    await expect(panel(page, "Tests").getByRole("link", { name: "main" })).toBeVisible();
  });

  test("creating an issue adds it to the list and calls gh issue create", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    const issues = panel(page, "Issues");
    await issues.getByRole("button", { name: "New issue" }).click();
    await issues.getByLabel("Title").fill("E2E created issue");
    await issues.getByLabel("Details (optional)").fill("made by the end-to-end test");
    await issues.getByRole("button", { name: "Create issue" }).click();
    await expect(issues.getByRole("status")).toContainText("Issue created");
    await expect(issues.getByRole("list", { name: "Open issues" })).toContainText("E2E created issue");
    const call = createCalls(dash).find((c) => c.args.includes("--title=E2E created issue"));
    expect(call?.args).toContain("--body=made by the end-to-end test");
    expect(dash.ghIssues().some((i) => i.title === "E2E created issue" && i.state === "OPEN")).toBe(true);
  });

  test("the kanban is read-only before the first import and says how to import", async ({ page }) => {
    const writesBefore = dashboard().ghCalls().filter((c) => ["create", "close", "reopen", "edit"].includes(c.args[1])).length;
    await open(page, dashboard());
    const kanban = panel(page, "Kanban");
    await expect(kanban.getByRole("note")).toContainText("npm run kanban:sync -- --dry-run");
    await expect(kanban.getByText("Tune the vote threshold")).toBeVisible();
    await expect(kanban.getByRole("checkbox")).toHaveCount(0);
    await expect(kanban.getByRole("button", { name: "Add task" })).toHaveCount(0);
    await expect(kanban.getByRole("button", { name: "Sync now" })).toHaveCount(0);
    await expect(kanban.getByText("Pick the detector split")).toBeVisible();
    const writesAfter = dashboard().ghCalls().filter((c) => ["create", "close", "reopen", "edit"].includes(c.args[1])).length;
    expect(writesAfter).toBe(writesBefore); // reading the board from the markdown made no gh write
  });

  test("Run tests streams lines, then shows the summary and the failing test; a second start is refused", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    const local = page.getByRole("region", { name: "Local run", exact: true });
    await local.getByRole("button", { name: "Run tests" }).click();
    const running = local.getByRole("button", { name: "Running…" });
    await expect(running).toBeDisabled();
    // a second start while running does not start a second run
    await running.click({ force: true });
    const second = await page.request.post(`${dash.baseUrl}/api/tests/run`, { headers: { "Content-Type": "application/json" } });
    expect(second.status()).toBe(409);
    const log = local.getByLabel("Test output");
    await expect(log).toContainText("collected 5 items");
    await expect(local.getByRole("list", { name: "Failing tests" })).toContainText("tests.test_hitfinder::test_vote_aggregation_breaks_ties", { timeout: 20_000 });
    await expect(local.getByRole("button", { name: "Run tests" })).toBeEnabled();
    await expect(local.getByText("4 passed", { exact: true })).toBeVisible();
    await expect(local.getByText("1 failed", { exact: true })).toBeVisible();
    await expect(log).toContainText("1 failed, 4 passed");
    expect(dash.pytestRuns()).toBe(1);
  });

  test("the light and dark toggle persists across a reload", async ({ page }) => {
    await open(page, dashboard());
    const html = page.locator("html");
    const toggle = page.getByRole("button", { name: "Light theme" });
    await toggle.click();
    await expect(html).toHaveAttribute("data-theme", "light");
    await page.reload();
    await expect(html).toHaveAttribute("data-theme", "light");
    await expect(page.getByRole("button", { name: "Light theme" })).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("button", { name: "Light theme" }).click();
    await expect(html).toHaveAttribute("data-theme", "dark");
    await page.reload();
    await expect(html).toHaveAttribute("data-theme", "dark");
  });

  for (const size of [{ width: 1440, height: 900 }, { width: 1024, height: 768 }]) {
    test(`no horizontal page scroll at ${size.width}x${size.height}`, async ({ page }) => {
      await page.setViewportSize(size);
      await open(page, dashboard());
      await expect(panel(page, "Kanban")).toBeVisible();
      await expect(panel(page, "Issues")).toBeVisible();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow).toBeLessThanOrEqual(0);
    });
  }
});

test.describe("empty states", () => {
  const dashboard = withDashboard({ seed: "empty", kanban: "# Nothing yet\n" });

  test("each panel says what would appear", async ({ page }) => {
    await open(page, dashboard());
    await expect(panel(page, "Latest pull request")).toContainText("No merged pull requests yet");
    await expect(panel(page, "Issues")).toContainText("No open issues");
    await expect(panel(page, "Tests")).toContainText("No CI runs found");
    await expect(panel(page, "Tests")).toContainText("No local run yet");
    await expect(panel(page, "Kanban")).toContainText("No tasks found in the kanban file yet");
  });
});

test.describe("error states", () => {
  const dashboard = withDashboard();

  test("a failing gh shows an error in each panel and Retry recovers", async ({ page }) => {
    const dash = dashboard();
    dash.failGh(["pr list", "issue list", "run list"]);
    await open(page, dash);
    for (const title of ["Latest pull request", "Issues"]) {
      const p = panel(page, title);
      await expect(p.getByRole("alert")).toContainText("GitHub CLI failed");
      await expect(p.getByRole("button", { name: "Retry" })).toBeVisible();
    }
    await expect(panel(page, "Tests").getByRole("alert").first()).toContainText("GitHub CLI failed");
    dash.clearGhFailures();
    await panel(page, "Issues").getByRole("button", { name: "Retry" }).click();
    await expect(panel(page, "Issues").getByRole("list", { name: "Open issues" })).toContainText("Eiger4M geometry looks off");
    await panel(page, "Latest pull request").getByRole("button", { name: "Retry" }).click();
    await expect(panel(page, "Latest pull request").getByRole("link", { name: "Add the status dashboard" })).toBeVisible();
  });
});

test.describe("kanban after the import", () => {
  const dashboard = withDashboard();
  const NEW_TASK = "E2E added task";

  test.describe.configure({ mode: "serial" });

  test("importing from the command line makes the board live", async ({ page }) => {
    const dash = dashboard();
    const dry = await dash.runCli(["--dry-run"]);
    expect(dry.code).toBe(0);
    expect(createCalls(dash)).toHaveLength(0);
    const real = await dash.runCli(["--yes"]);
    expect(real.stderr).toBe("");
    expect(real.code).toBe(0);
    expect(createCalls(dash)).toHaveLength(2);
    expect(dash.kanbanText()).toMatch(/Tune the vote threshold <!-- gh:#\d+ -->/);
    await open(page, dash);
    await expect(column(page, "In progress")).toContainText("Tune the vote threshold");
    await expect(column(page, "Done")).toContainText("Pick the detector split");
    await expect(panel(page, "Kanban").getByRole("button", { name: "Add task" })).toBeVisible();
  });

  test("adding a task puts it in Todo, in the markdown under the inbox heading, and creates an issue", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    const before = createCalls(dash).length;
    await panel(page, "Kanban").getByRole("textbox", { name: "Task", exact: true }).fill(NEW_TASK);
    await panel(page, "Kanban").getByRole("button", { name: "Add task" }).click();
    await expect(column(page, "Todo")).toContainText(NEW_TASK);
    const calls = createCalls(dash);
    expect(calls).toHaveLength(before + 1);
    expect(calls.at(-1)!.args).toEqual(expect.arrayContaining([`--title=${NEW_TASK}`, "--label=task", "--label=status:todo"]));
    await expect.poll(() => dash.kanbanText()).toMatch(new RegExp(`## Inbox \\(added via dashboard\\)\\n+- \\[ \\] ${NEW_TASK} <!-- gh:#\\d+ -->`));
  });

  test("completing a task strikes it through, ticks the markdown and closes the issue; reopening reverses all three", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    const box = panel(page, "Kanban").getByRole("checkbox", { name: `Mark ${NEW_TASK} done` });
    await box.check();
    await expect(column(page, "Done").locator("s", { hasText: NEW_TASK })).toBeVisible();
    await expect.poll(() => dash.kanbanText()).toMatch(new RegExp(`- \\[x\\] ${NEW_TASK} <!-- gh:#\\d+ -->`));
    await expect.poll(() => dash.ghIssues().find((i) => i.title === NEW_TASK)?.state).toBe("CLOSED");

    await panel(page, "Kanban").getByRole("checkbox", { name: `Mark ${NEW_TASK} done` }).uncheck();
    await expect(column(page, "Todo")).toContainText(NEW_TASK);
    await expect(column(page, "Todo").locator("s", { hasText: NEW_TASK })).toHaveCount(0);
    await expect.poll(() => dash.kanbanText()).toMatch(new RegExp(`- \\[ \\] ${NEW_TASK} <!-- gh:#\\d+ -->`));
    await expect.poll(() => dash.ghIssues().find((i) => i.title === NEW_TASK)?.state).toBe("OPEN");
  });

  test("the add-task form and the move control work with the keyboard alone", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    const title = "Keyboard only task";
    const input = panel(page, "Kanban").getByRole("textbox", { name: "Task", exact: true });
    // Reach the field by tabbing from the top of the page.
    await page.locator("body").press("Tab");
    for (let i = 0; i < 80; i++) {
      if (await input.evaluate((el) => el === document.activeElement)) break;
      await page.keyboard.press("Tab");
    }
    await expect(input).toBeFocused();
    await page.keyboard.type(title);
    await page.keyboard.press("Enter");
    await expect(column(page, "Todo")).toContainText(title);
    // Move it with the select: focus, then arrow down to "In progress".
    const move = panel(page, "Kanban").getByLabel(`Move ${title}`);
    await move.focus();
    await page.keyboard.press("ArrowDown");
    await expect(column(page, "In progress")).toContainText(title);
    await expect(move).toBeFocused();
    await expect.poll(() => dash.ghIssues().find((i) => i.title === title)?.labels ?? []).toContain("status:in-progress");
  });

  test("an unreachable GitHub shows a sync problem on the board, and Sync now recovers", async ({ page }) => {
    // The board is built from the list the last sync fetched, so an unreachable GitHub shows up
    // as a sync problem on that board rather than as a failed board read.
    const dash = dashboard();
    dash.failGh(["issue list"]);
    await open(page, dash);
    await panel(page, "Kanban").getByRole("button", { name: "Sync now" }).click();
    await expect(panel(page, "Kanban").getByText("Sync problem")).toBeVisible();
    dash.clearGhFailures();
    await panel(page, "Kanban").getByRole("button", { name: "Sync now" }).click();
    await expect(panel(page, "Kanban").getByText("Sync problem")).toHaveCount(0);
    await expect(column(page, "Todo")).toBeVisible();
  });

  test("the markdown copy stayed inside the temp directory", async () => {
    const dash = dashboard();
    expect(path.relative(dash.dir, dash.kanbanPath).startsWith("..")).toBe(false);
    expect(fs.existsSync(path.join(dash.stateDir, "kanban-sync.json"))).toBe(true);
  });
});

test.describe("runs panel", () => {
  const dashboard = withDashboard();
  const PROVISIONAL = "vits16-mae-finetune-v2-fold2-seed42";
  const WARNING = "training did not finish: wandb state crashed";

  test("lists the fixture runs, flags the provisional one, and skips non-runs and symlinks", async ({ page }) => {
    await open(page, dashboard());
    const runs = panel(page, "Runs");
    await expect(runs.getByRole("table", { name: "Training runs" })).toBeVisible();
    await expect(runs.getByRole("row")).toHaveCount(10); // header + 9 runs
    const provisional = runs.getByRole("row").filter({ hasText: PROVISIONAL });
    await expect(provisional).toContainText("Provisional");
    await expect(provisional).toContainText("Re-evaluated");
    const complete = runs.getByRole("row").filter({ hasText: "resnet18-asymmetric-v2-fold1-seed42" });
    await expect(complete).toContainText("Complete");
    await expect(complete).toContainText("agipd");
    // Epochs: read from the real checkpoint here, from the training check there
    await expect(complete).toContainText("best 187");
    await expect(provisional).toContainText("187 / 200");
    await expect(runs.getByRole("row").filter({ hasText: "vits16-mae-finetune-fold2-seed42-v2" })).toContainText("Archived v1");
    await expect(runs.getByRole("row").filter({ hasText: "mae-vits16-v2-fold1-seed42" })).toContainText("epoch 100");
    for (const hidden of ["asymmetric_splits", "empty-dir", "escape-link"]) await expect(runs).not.toContainText(hidden);
  });

  test("the re-evaluation is a second tab and never replaces the fold's result", async ({ page }) => {
    await open(page, dashboard());
    const runs = panel(page, "Runs");
    await runs.getByRole("button", { name: new RegExp(PROVISIONAL) }).click();
    await expect(runs.getByText(WARNING)).toBeVisible();
    await runs.getByRole("tab", { name: "Re-evaluation" }).click();
    await expect(runs.getByRole("tabpanel")).toContainText("Re-evaluation, not the fold's result");
    await expect(runs.getByText(WARNING)).toHaveCount(0);
    await runs.getByRole("tab", { name: "Result" }).click();
    await expect(runs.getByText(WARNING)).toBeVisible();
  });

  test("sorting by Status puts the provisional fold first and marks the column; Updated sorts newest first", async ({ page }) => {
    await open(page, dashboard());
    const runs = panel(page, "Runs");
    const status = runs.getByRole("columnheader", { name: /^Status/ });
    await status.getByRole("button").click();
    await expect(status).toHaveAttribute("aria-sort", "ascending");
    await expect(runs.getByRole("row").nth(1)).toContainText("Provisional");
    const updated = runs.getByRole("columnheader", { name: /^Updated/ });
    await updated.getByRole("button").click();
    await expect(updated).toHaveAttribute("aria-sort", "descending");
    await expect(status).toHaveAttribute("aria-sort", "none");
    await expect(runs.getByRole("status")).toContainText("Sorted by Updated, newest first");
  });

  test("a corrupt results.json shows a fixed message on its row and nothing else breaks", async ({ page }) => {
    await open(page, dashboard());
    const runs = panel(page, "Runs");
    await runs.getByRole("button", { name: /resnet50-asymmetric-v2-fold1-seed42/ }).click();
    await expect(runs.getByRole("alert")).toHaveText("results.json: not valid JSON");
    await expect(runs.getByRole("alert")).not.toContainText("/");
  });

  for (const [width, height] of [[1440, 900], [1024, 768]]) {
    test(`no horizontal page scroll at ${width}x${height} with a run expanded`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await open(page, dashboard());
      await panel(page, "Runs").getByRole("button", { name: new RegExp(PROVISIONAL) }).click();
      await expect(panel(page, "Runs").getByRole("tabpanel")).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
    });
  }
});

test.describe("runs panel without a checkpoints directory", () => {
  const dashboard = withDashboard({ checkpoints: false });
  test("explains that the panel needs the same filesystem", async ({ page }) => {
    await open(page, dashboard());
    await expect(panel(page, "Runs")).toContainText("No checkpoints directory on this machine");
  });
});

test.describe("agents and approvals", () => {
  const dashboard = withDashboard();
  const agents = (page: Page) => panel(page, "Agents");
  const approvals = (page: Page) => panel(page, "Approvals");

  async function start(page: Page, name: string, task: string): Promise<void> {
    await agents(page).getByRole("button", { name: "New agent" }).click();
    await page.getByLabel("Name", { exact: true }).fill(name);
    await page.getByLabel("Task", { exact: true }).fill(task);
    await page.getByRole("button", { name: "Start agent" }).click();
    await expect(agents(page).getByRole("button", { name: new RegExp(name) })).toBeVisible();
  }
  async function stopAll(page: Page, dash: Dashboard, names: string[]): Promise<void> {
    for (const n of names) await page.request.post(`${dash.baseUrl}/api/agents/${n}/stop`, { data: {} });
  }
  const row = (page: Page, name: string) => agents(page).getByRole("listitem").filter({ hasText: name });

  test("starting an agent shows it working, then idle, with its log", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    await expect(agents(page).getByText(/No agents yet/)).toBeVisible();
    await start(page, "tidy-notes", "tidy the notes");
    await expect(row(page, "tidy-notes").getByText("Idle")).toBeVisible();
    await row(page, "tidy-notes").getByRole("button", { name: /tidy-notes/ }).click();
    const log = page.getByRole("log", { name: /tidy-notes/ });
    await expect(log).toContainText("Working on: tidy the notes");
    await stopAll(page, dash, ["tidy-notes"]);
    await expect(row(page, "tidy-notes").getByText("Stopped", { exact: true })).toBeVisible();
  });

  test("a gated command waits on a card with the exact command; approving continues the agent", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    await start(page, "pusher-ok", "please push the branch");
    const card = approvals(page).getByRole("article", { name: /pusher-ok/ });
    await expect(card.getByText("git push origin HEAD")).toBeVisible();
    await expect(row(page, "pusher-ok").getByText("Waiting for approval")).toBeVisible();
    await expect(page.getByRole("link", { name: /1 waiting for approval/ })).toBeVisible();
    await card.getByRole("button", { name: "Approve" }).click();
    await expect(approvals(page).getByText(/Nothing is waiting/)).toBeVisible();
    await expect(row(page, "pusher-ok").getByText("Idle")).toBeVisible();
    await expect(page.getByRole("link", { name: /waiting for approval/ })).toHaveCount(0);
    await row(page, "pusher-ok").getByRole("button", { name: /pusher-ok/ }).click();
    await expect(page.getByRole("log", { name: /pusher-ok/ })).toContainText("Approval allowed for Bash");
    await stopAll(page, dash, ["pusher-ok"]);
  });

  test("denying tells the agent, which carries on", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    await start(page, "pusher-no", "please push the branch");
    await approvals(page).getByRole("button", { name: "Deny" }).click();
    await expect(row(page, "pusher-no").getByText("Idle")).toBeVisible();
    await row(page, "pusher-no").getByRole("button", { name: /pusher-no/ }).click();
    const log = page.getByRole("log", { name: /pusher-no/ });
    await expect(log).toContainText("Approval denied for Bash");
    await expect(log).toContainText("Denied");
    await stopAll(page, dash, ["pusher-no"]);
  });

  test("a pending approval survives a reload, and stopping the agent removes its card", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    await start(page, "pusher-wait", "please push the branch");
    await expect(approvals(page).getByRole("article")).toHaveCount(1);
    await page.reload();
    await expect(approvals(page).getByRole("article", { name: /pusher-wait/ })).toBeVisible();
    await row(page, "pusher-wait").getByRole("button", { name: /pusher-wait/ }).click();
    await page.getByRole("button", { name: "Stop", exact: true }).click();
    await expect(approvals(page).getByText(/Nothing is waiting/)).toBeVisible();
    await expect(row(page, "pusher-wait").getByText("Stopped")).toBeVisible();
  });

  test("an approval can be made with the keyboard alone", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    await start(page, "pusher-key", "please push the branch");
    const approve = approvals(page).getByRole("button", { name: "Approve" });
    await approve.focus();
    await expect(approve).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(approvals(page).getByText(/Nothing is waiting/)).toBeVisible();
    await stopAll(page, dash, ["pusher-key"]);
  });

  test("the cap is enforced with a plain message", async ({ page }) => {
    const dash = dashboard();
    await open(page, dash);
    for (const n of ["cap-one", "cap-two", "cap-three"]) await start(page, n, "work");
    await agents(page).getByRole("button", { name: "New agent" }).click();
    await page.getByLabel("Name", { exact: true }).fill("cap-four");
    await page.getByLabel("Task", { exact: true }).fill("work");
    await page.getByRole("button", { name: "Start agent" }).click();
    await expect(page.getByRole("alert").filter({ hasText: "3 agents are already running" })).toBeVisible();
    await stopAll(page, dash, ["cap-one", "cap-two", "cap-three"]);
  });

  for (const size of [{ width: 1440, height: 900 }, { width: 1024, height: 768 }]) {
    test(`no horizontal scroll with an approval card open at ${size.width}x${size.height}`, async ({ page }) => {
      const dash = dashboard();
      await page.setViewportSize(size);
      await open(page, dash);
      await start(page, `layout-${size.width}`, "please push the branch");
      await expect(approvals(page).getByRole("article")).toHaveCount(1);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow).toBeLessThanOrEqual(0);
      await stopAll(page, dash, [`layout-${size.width}`]);
    });
  }
});
