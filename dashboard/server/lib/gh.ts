import type { Config } from "../config";
import type { Runner } from "../runner";

const GH_PUBLIC_MESSAGE = "GitHub CLI failed";
const GH_PUBLIC_INVALID_JSON = "GitHub CLI returned invalid JSON";
const STDERR_MAX_CHARS = 500;

export class GhError extends Error {
  constructor(
    message: string,
    public args: string[],
    public code: number,
    public publicMessage: string = GH_PUBLIC_MESSAGE,
    /** gh's own stderr, without the command line (which carries user text such as an issue title). */
    public stderr: string = "",
  ) {
    super(message);
    this.name = "GhError";
  }
}

/**
 * Pass a value to gh as one `--name=value` argv element so a value that starts
 * with `-` (for example "--web") can never be parsed as a separate flag.
 */
export function flagArg(name: string, value: string): string {
  return `--${name}=${value}`;
}

export interface Gh {
  json<T>(args: string[]): Promise<T>;
  text(args: string[]): Promise<string>;
}

export function createGh(runner: Runner, cfg: Pick<Config, "ghBin" | "repoRoot">): Gh {
  async function run(args: string[]): Promise<string> {
    const r = await runner(cfg.ghBin, args, { cwd: cfg.repoRoot });
    if (r.code !== 0) {
      throw new GhError(`gh ${args.join(" ")} failed (exit ${r.code}): ${r.stderr.trim().slice(0, STDERR_MAX_CHARS)}`, args, r.code, GH_PUBLIC_MESSAGE, r.stderr);
    }
    return r.stdout;
  }
  return {
    async json<T>(args: string[]): Promise<T> {
      const out = await run(args);
      try {
        return JSON.parse(out) as T;
      } catch {
        throw new GhError(`gh ${args.join(" ")} returned invalid JSON`, args, -1, GH_PUBLIC_INVALID_JSON);
      }
    },
    async text(args: string[]): Promise<string> {
      return (await run(args)).trim();
    },
  };
}

const EXIT_TIMEOUT = 124; // execRunner: the process was killed after the time limit
const EXIT_SPAWN_FAILURE = 127; // execRunner: gh could not be started
/** gh stderr that means every further call will fail too: authentication, quota, outage, network. */
const GLOBAL_STDERR =
  /\bHTTP\s*(?:401|403|429|5\d\d)\b|bad credentials|gh auth login|not logged in|authentication (?:failed|required)|rate limit|abuse detection|submitted too quickly|error connecting to|could not resolve host|no such host|network is unreachable|connection (?:refused|reset)|timed? ?out|\bENOTFOUND\b|\bECONN\w*|\bETIMEDOUT\b|\bENOENT\b|\bEAI_AGAIN\b/i;

/**
 * Decides whether a gh failure is specific to one item or would hit every item. Looks only at gh's
 * stderr and exit code, never at the message, which includes the arguments and so the item's own text.
 */
export function classifyGhFailure(err: unknown): "global" | "item" {
  if (!(err instanceof GhError)) return "item";
  if (err.code === EXIT_TIMEOUT || err.code === EXIT_SPAWN_FAILURE) return "global";
  return GLOBAL_STDERR.test(err.stderr) ? "global" : "item";
}
