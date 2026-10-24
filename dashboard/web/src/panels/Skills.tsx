import { useState } from "react";
import { PanelFrame } from "../components/PanelFrame";
import { panelState } from "../components/bits";
import { ApiError, apiSend } from "../lib/api";
import { relTime } from "../lib/time";
import { useResource } from "../lib/useResource";
import type { SkillSummary, SkillsResponse } from "../../../shared/types";
import { SkillArt } from "./SkillCartoons";
import "./Skills.css";

const ARGS_MAX = 500;
const SUFFIX_LEN = 4;
const SUFFIX_RADIX = 36;
/** Skill cards change only when a transcript does, so a slow poll is plenty. */
const SKILLS_REFRESH_MS = 60_000;

const message = (err: unknown, fallback: string) => (err instanceof ApiError ? err.message : fallback);
const agentName = (skill: string) => `skill-${skill}-${Date.now().toString(SUFFIX_RADIX).slice(-SUFFIX_LEN)}`;

function SkillCard({ s, open, onToggle }: { s: SkillSummary; open: boolean; onToggle: () => void }) {
  const [args, setArgs] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState<string | null>(null);
  const detailId = `skill-${s.name}`;

  async function launch() {
    if (busy) return;
    setBusy(true);
    setError(null);
    const name = agentName(s.name);
    try {
      await apiSend("POST", "/api/agents", { name, task: `Run the project skill /${s.name}${args.trim() ? ` ${args.trim()}` : ""}` });
      setStarted(name);
      setConfirming(false);
    } catch (err) {
      setError(message(err, "Could not start the agent."));
    } finally {
      setBusy(false);
    }
  }

  const onRun = () => {
    if (s.confirmRequired && !confirming) return setConfirming(true);
    void launch();
  };

  return (
    <li className="skill" data-open={open}>
      <button type="button" className="skill__card" aria-expanded={open} aria-controls={detailId} onClick={onToggle}>
        <SkillArt name={s.name} />
        <span className="skill__name mono">{s.name}</span>
        <span className="skill__when dim">{s.lastRun ? `ran ${relTime(s.lastRun)}` : "never ran"}</span>
      </button>
      {open && (
        <div id={detailId} className="skill__detail" role="region" aria-label={`${s.name} details`}>
          <p className="skill__desc">{s.description || "No description."}</p>
          <label className="field__label" htmlFor={`${detailId}-args`}>Arguments (optional)</label>
          <input id={`${detailId}-args`} className="field mono" maxLength={ARGS_MAX} value={args} onChange={(e) => setArgs(e.target.value)} autoComplete="off" spellCheck={false} />
          {error && <p className="approval__error" role="alert">{error}</p>}
          {started && <p className="dim" role="status">Started agent <span className="mono">{started}</span>.</p>}
          <div className="row row--wrap">
            {confirming ? (
              <>
                <span className="skill__warn">Start /{s.name} as a new agent?</span>
                <button type="button" className="btn btn--primary" disabled={busy} onClick={onRun}>Yes, run</button>
                <button type="button" className="btn btn--quiet" disabled={busy} onClick={() => setConfirming(false)}>Cancel</button>
              </>
            ) : (
              <button type="button" className="btn btn--primary" disabled={busy} onClick={onRun}>Run</button>
            )}
          </div>
        </div>
      )}
    </li>
  );
}

/** The project's own skills as square cards: click to see details, then Run to hand one to a new agent. */
export function Skills() {
  const res = useResource<SkillsResponse>("skills", "/api/skills", (d) => d.skills.length === 0, SKILLS_REFRESH_MS);
  const [open, setOpen] = useState<string | null>(null);
  const skills = res.data?.skills ?? [];
  return (
    <PanelFrame title="Skills" count={skills.length} state={panelState(res)} error={res.error ?? undefined} onRetry={res.reload}>
      <ul className="skills">
        {skills.map((s) => (
          <SkillCard key={s.name} s={s} open={open === s.name} onToggle={() => setOpen(open === s.name ? null : s.name)} />
        ))}
      </ul>
    </PanelFrame>
  );
}
