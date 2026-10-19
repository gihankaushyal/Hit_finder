export function RightRail() {
  return (
    <aside className="rail" aria-label="Agents and chief of staff">
      <section className="panel" aria-labelledby="rail-agents">
        <header className="panel__head">
          <h2 id="rail-agents" className="panel__title">Agents</h2>
        </header>
        <div className="panel__body"><p className="dim">Arrives in milestone 2.</p></div>
      </section>
      <section className="panel" aria-labelledby="rail-chief">
        <header className="panel__head">
          <h2 id="rail-chief" className="panel__title">Chief of staff</h2>
        </header>
        <div className="panel__body"><p className="dim">Arrives in milestone 2.</p></div>
      </section>
    </aside>
  );
}
