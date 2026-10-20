import { Agents } from "./Agents";
import { Approvals } from "./Approvals";

export function RightRail() {
  return (
    <aside className="rail" aria-label="Agents and chief of staff">
      <Approvals />
      <Agents />
      <section className="panel" aria-labelledby="rail-chief">
        <header className="panel__head">
          <h2 id="rail-chief" className="panel__title">Chief of staff</h2>
        </header>
        <div className="panel__body"><p className="dim">Arrives in the next milestone.</p></div>
      </section>
    </aside>
  );
}
