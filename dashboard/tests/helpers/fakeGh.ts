import type { Gh } from "../../server/lib/gh";

export interface FakeIssue {
  number: number;
  title: string;
  state: "OPEN" | "CLOSED";
  labels: string[];
  body: string;
  url: string;
  updatedAt: string;
}

/** In-memory stand-in for the gh CLI. Every `text` call is a mutation and is recorded in `writes`. */
export class FakeGh implements Gh {
  issues = new Map<number, FakeIssue>();
  writes: string[][] = [];
  jsonCalls: string[][] = [];
  next = 1;
  creates = 0;
  /** Throw on the Nth create (1-based). */
  failCreateAt: number | null = null;
  failClose = false;
  /** Called after each successful create, with the new issue number. */
  onCreate: (n: number) => void = () => {};
  /** Output override for `issue create`. */
  createOutput: ((n: number) => string) | null = null;

  seed(partial: Partial<FakeIssue> & { title: string }): FakeIssue {
    const n = partial.number ?? this.next++;
    if (n >= this.next) this.next = n + 1;
    const issue: FakeIssue = {
      number: n, state: "OPEN", labels: ["task"], body: "", url: `https://github.com/o/r/issues/${n}`,
      updatedAt: "2026-10-07T00:00:00Z", ...partial,
    };
    this.issues.set(n, issue);
    return issue;
  }

  async json<T>(args: string[]): Promise<T> {
    this.jsonCalls.push(args);
    return [...this.issues.values()].map((i) => ({ ...i, labels: i.labels.map((name) => ({ name })) })) as T;
  }

  async text(args: string[]): Promise<string> {
    this.writes.push(args);
    const [group, verb] = args;
    const flag = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
    if (group === "label") return "";
    if (verb === "create") {
      this.creates++;
      if (this.failCreateAt === this.creates) throw new Error("gh create failed");
      const issue = this.seed({
        title: flag("title") ?? "",
        body: flag("body") ?? "",
        labels: args.filter((a) => a.startsWith("--label=")).map((a) => a.slice(8)),
      });
      this.onCreate(issue.number);
      return this.createOutput ? this.createOutput(issue.number) : issue.url;
    }
    const issue = this.issues.get(Number(args[2]));
    if (!issue) throw new Error("no such issue");
    if (verb === "close") {
      if (this.failClose) throw new Error("gh close failed");
      issue.state = "CLOSED";
    } else if (verb === "reopen") issue.state = "OPEN";
    else if (verb === "edit") issue.body = flag("body") ?? issue.body;
    return "";
  }

  count(verb: string): number {
    return this.writes.filter((w) => w[1] === verb && w[0] === "issue").length;
  }
}
