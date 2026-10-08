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
      throw new GhError(`gh ${args.join(" ")} failed (exit ${r.code}): ${r.stderr.trim().slice(0, STDERR_MAX_CHARS)}`, args, r.code);
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
