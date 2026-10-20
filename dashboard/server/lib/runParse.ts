import type { MetricBlock, RunInference, RunKind, RunResult, RunStatus, RunTraining, TrainingCheckStatus } from "../../shared/types";

// Copies of the patterns in src/training/run_naming.py. The dashboard never imports src/.
const SUFFIX = "(?:-[a-z0-9-]+)?";
const TAIL = `-fold\\d+-seed\\d+${SUFFIX}$`;
const PATTERNS: [RunKind, RegExp][] = [
  ["track1", new RegExp(`^[a-z0-9]+-asymmetric-v(\\d+)${TAIL}`)],
  ["ssl-pretrain", new RegExp(`^mae-[a-z0-9]+-v(\\d+)${TAIL}`)],
  ["ssl-finetune", new RegExp(`^[a-z0-9]+-mae-finetune-v(\\d+)${TAIL}`)],
  ["ssl-probe", new RegExp(`^[a-z0-9]+-mae-probe-v(\\d+)${TAIL}`)],
];

const MAX_METRICS = 32;
const MAX_WARNINGS = 20;
const MAX_WARNING_CHARS = 500;
const CHECK_STATUSES: readonly string[] = ["complete", "incomplete", "unknown"];

/** Names that do not follow the naming rules are shown as `legacy`, never hidden. */
export function classifyName(name: string): { kind: RunKind; generation: number | null } {
  for (const [kind, re] of PATTERNS) {
    const m = re.exec(name);
    if (m) return { kind, generation: Number(m[1]) };
  }
  return { kind: "legacy", generation: null };
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);

// Keys that would change a plain object's prototype or shadow its constructor when assigned.
const UNSAFE_KEYS = new Set(["__proto__", "constructor", "prototype"]);

function metricBlock(v: unknown): MetricBlock | null {
  if (!isObj(v)) return null;
  const out: MetricBlock = {};
  for (const [k, val] of Object.entries(v).filter(([key]) => !UNSAFE_KEYS.has(key)).slice(0, MAX_METRICS)) out[k] = num(val);
  return out;
}

function parseInference(v: unknown): RunInference | null {
  if (!isObj(v)) return null;
  const tc = isObj(v.training_check) ? v.training_check : null;
  const status = tc && typeof tc.status === "string" && CHECK_STATUSES.includes(tc.status) ? (tc.status as TrainingCheckStatus) : null;
  const warnings = tc && Array.isArray(tc.warnings)
    ? tc.warnings.filter((w): w is string => typeof w === "string").slice(0, MAX_WARNINGS).map((w) => w.slice(0, MAX_WARNING_CHARS))
    : [];
  return {
    aggregation: str(v.aggregation),
    checkpointEpoch: num(v.checkpoint_epoch),
    evaluatedAt: str(v.evaluated_at),
    trainingStatus: status,
    configuredEpochs: tc ? num(tc.configured_epochs) : null,
    lastLoggedEpoch: tc ? num(tc.last_logged_epoch) : null,
    wandbState: tc ? str(tc.wandb_state) : null,
    warnings,
  };
}

const NON_FINITE = /-?(?:NaN|Infinity)(?![A-Za-z0-9_])/y;

/**
 * Python's json module writes NaN, Infinity and -Infinity, which JSON.parse rejects. Replaces those bare
 * tokens with null, leaving anything inside a string untouched.
 */
export function replaceNonFiniteTokens(text: string): string {
  if (!/NaN|Infinity/.test(text)) return text;
  let out = "";
  let inString = false;
  for (let i = 0; i < text.length; ) {
    const c = text[i];
    if (inString) {
      if (c === "\\") {
        out += text.slice(i, i + 2);
        i += 2;
        continue;
      }
      if (c === '"') inString = false;
      out += c;
      i++;
      continue;
    }
    if (c === '"') {
      inString = true;
    } else if (c === "N" || c === "I" || c === "-") {
      NON_FINITE.lastIndex = i;
      const m = NON_FINITE.exec(text);
      if (m && !/[A-Za-z0-9_]/.test(text[i - 1] ?? "")) {
        out += "null";
        i += m[0].length;
        continue;
      }
    }
    out += c;
    i++;
  }
  return out;
}

/** The block the training code writes into results.json; any field may be missing. */
function parseTraining(v: unknown): RunTraining | null {
  if (!isObj(v)) return null;
  return {
    configuredEpochs: num(v.configured_epochs),
    lastEpoch: num(v.last_epoch),
    bestEpoch: num(v.best_epoch),
    earlyStopped: typeof v.early_stopped === "boolean" ? v.early_stopped : null,
  };
}

/** Throws an Error whose message is a fixed string safe to show a client. */
export function parseResult(text: string): RunResult {
  let raw: unknown;
  try {
    raw = JSON.parse(replaceNonFiniteTokens(text));
  } catch {
    throw new Error("not valid JSON");
  }
  if (!isObj(raw)) throw new Error("not a JSON object");
  return {
    foldId: num(raw.fold_id),
    testDetector: str(raw.test_detector),
    inferenceThreshold: num(raw.inference_threshold),
    cross: metricBlock(raw.cross),
    inDomain: metricBlock(raw.in_domain),
    track: str(raw.track),
    probe: typeof raw.probe === "boolean" ? raw.probe : null,
    pretrainEpoch: num(raw.pretrain_epoch),
    inference: parseInference(raw.inference),
    training: parseTraining(raw.training),
  };
}

export function deriveStatus(
  kind: RunKind,
  files: { resultsJson: boolean; anyCheckpoint: boolean; pretrainShaped: boolean },
  result: RunResult | null,
  parseFailed: boolean,
): RunStatus {
  if (kind === "ssl-pretrain" || files.pretrainShaped) return "pretrain";
  if (!files.resultsJson) return "no-results";
  if (parseFailed || result === null) return "unknown";
  if (result.inference === null) return "complete";
  switch (result.inference.trainingStatus) {
    case "complete": return "complete";
    case "incomplete": return "provisional";
    default: return "unknown";
  }
}
