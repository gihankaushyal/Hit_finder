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
    const running = local.getByRole("button", { name: "Running" });
    await expect(running).toBeDisabled();
    // a second start while running does not start a second run
    await running.click({ force: true });
    const second = await page.request.post(`${dash.baseUrl}/api/tests/run`);
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

  test("an unreachable GitHub shows the error and the board recovers after Retry", async ({ page }) => {
    const dash = dashboard();
    dash.failGh(["issue list"]);
    await open(page, dash);
    await expect(panel(page, "Kanban").getByRole("alert")).toContainText("GitHub CLI failed");
    dash.clearGhFailures();
    await panel(page, "Kanban").getByRole("button", { name: "Retry" }).click();
    await expect(column(page, "Todo")).toBeVisible();
  });

  test("the markdown copy stayed inside the temp directory", async () => {
    const dash = dashboard();
    expect(path.relative(dash.dir, dash.kanbanPath).startsWith("..")).toBe(false);
    expect(fs.existsSync(path.join(dash.stateDir, "kanban-sync.json"))).toBe(true);
  });
});
