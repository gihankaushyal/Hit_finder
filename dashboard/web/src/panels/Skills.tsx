import { useState } from "react";
import { PanelFrame } from "../components/PanelFrame";
import { panelState } from "../components/bits";
import { apiSend, errorMessage } from "../lib/api";
import { relTime } from "../lib/time";
import { useResource } from "../lib/useResource";
import type { AgentSummary, SkillSummary, SkillsResponse } from "../../../shared/types";
import { SkillArt } from "./SkillCartoons";
import "./Skills.css";

const ARGS_MAX = 500;
/** Skill cards change only when a transcript does, so a slow poll is plenty. */
const SKILLS_REFRESH_MS = 60_000;
const DETAIL_ID = "skill-detail";


/** Description, arguments and Run for the open skill. Keyed by skill, so switching cards starts a fresh form. */
function SkillDetail({ s, onStarted }: { s: SkillSummary; onStarted: () => void }) {
  const [args, setArgs] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState<string | null>(null);

  async function launch(confirmed: boolean) {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const agent = await apiSend<AgentSummary>("POST", `/api/skills/${encodeURIComponent(s.name)}/run`, { args, confirmed });
      setStarted(agent.name);
      setConfirming(false);
      onStarted(); // the card's "ran" label should show this launch now, not at the next poll
    } catch (err) {
      setError(errorMessage(err, "Could not start the agent."));
    } finally {
      setBusy(false);
    }
  }

  const onRun = () => {
    if (s.confirmRequired && !confirming) return setConfirming(true);
    void launch(confirming);
  };

  return (
    <div id={DETAIL_ID} className="skill__detail" role="region" aria-label={`${s.name} details`}>
      <p className="skill__desc">{s.description || "No description."}</p>
      <label className="field__label" htmlFor={`${DETAIL_ID}-args`}>Arguments (optional)</label>
      <input id={`${DETAIL_ID}-args`} className="field mono" maxLength={ARGS_MAX} value={args} onChange={(e) => setArgs(e.target.value)} autoComplete="off" spellCheck={false} />
      {error && <p className="approval__error" role="alert">{error}</p>}
      {started && <p className="dim" role="status">Started agent <span className="mono">{started}</span>.</p>}
      <div className="row row--wrap">
        {confirming ? (
          <>
            <span className="skill__warn">Start {s.name} as a new agent?</span>
            <button type="button" className="btn btn--primary" disabled={busy} onClick={onRun}>Yes, run</button>
            <button type="button" className="btn btn--quiet" disabled={busy} onClick={() => setConfirming(false)}>Cancel</button>
          </>
        ) : (
          <button type="button" className="btn btn--primary" disabled={busy} onClick={onRun}>Run</button>
        )}
      </div>
    </div>
  );
}

/** The project's own skills as square cards: click one to see details, then Run to hand it to a new agent. */
export function Skills() {
  const res = useResource<SkillsResponse>("skills", "/api/skills", (d) => d.skills.length === 0, SKILLS_REFRESH_MS);
  const [open, setOpen] = useState<string | null>(null);
  const skills = res.data?.skills ?? [];
  const current = skills.find((s) => s.name === open);
  return (
    <PanelFrame title="Skills" count={skills.length} state={panelState(res)} error={res.error ?? undefined} onRetry={res.reload}>
      <ul className="skills">
        {skills.map((s) => (
          <li key={s.name}>
            <button type="button" className="skill__card" aria-expanded={open === s.name} aria-controls={open === s.name ? DETAIL_ID : undefined} onClick={() => setOpen(open === s.name ? null : s.name)}>
              <SkillArt art={s.art} />
              <span className="skill__name mono">{s.name}</span>
              <span className="skill__when dim">{s.lastRun ? `ran ${relTime(s.lastRun)}` : "never ran"}</span>
            </button>
          </li>
        ))}
      </ul>
      {current && <SkillDetail key={current.name} s={current} onStarted={res.reload} />}
    </PanelFrame>
  );
}
