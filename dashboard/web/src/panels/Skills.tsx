import { useState } from "react";
import { PanelFrame } from "../components/PanelFrame";
import { panelState } from "../components/bits";
import { relTime } from "../lib/time";
import { useResource } from "../lib/useResource";
import type { SkillSummary, SkillsResponse } from "../../../shared/types";
import { SkillArt } from "./SkillCartoons";
import "./Skills.css";

/** Skill cards change only when a transcript does, so a slow poll is plenty. */
const SKILLS_REFRESH_MS = 60_000;
const DETAIL_ID = "skill-detail";


/** Description and where to run the open skill. Skills run from a Claude Code session, not from here. */
function SkillDetail({ s }: { s: SkillSummary }) {
  return (
    <div id={DETAIL_ID} className="skill__detail" role="region" aria-label={`${s.name} details`}>
      <p className="skill__desc">{s.description || "No description."}</p>
      <p className="dim">
        Run it from Claude Code: <span className="mono">/{s.name}</span>
      </p>
    </div>
  );
}

/** The project's own skills as square cards: click one to see what it does and how to run it. */
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
      {current && <SkillDetail key={current.name} s={current} />}
    </PanelFrame>
  );
}
