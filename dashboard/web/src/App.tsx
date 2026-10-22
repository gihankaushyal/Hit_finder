import { useCallback, useRef, useState } from "react";
import { DEFAULT_TAB, DashboardNav, asTab, type TabId } from "./components/DashboardNav";
import { RightDrawer } from "./components/RightDrawer";
import { useQueryParam } from "./lib/useQueryState";
import { useResource } from "./lib/useResource";
import { Issues } from "./panels/Issues";
import { KanbanBoard } from "./panels/KanbanBoard";
import { PipelineCard } from "./panels/PipelineCard";
import { PrCard } from "./panels/PrCard";
import { RunsTable } from "./panels/RunsTable";
import { TerminalDrawer } from "./panels/TerminalDrawer";
import "./layout.css";

function Unauthorized({ message }: { message: string }) {
  return (
    <main className="gate" id="main">
      <h1>Open the tokenised URL again</h1>
      <p role="alert">{message}</p>
      <p className="dim">The server prints it on start, as an address ending in <span className="mono">?token=</span> followed by a long code.</p>
    </main>
  );
}

function Overview() {
  return (
    <div className="overview">
      <div className="overview__top">
        <PrCard />
        <PipelineCard />
        <div className="overview__kanban"><KanbanBoard /></div>
      </div>
      <RunsTable />
    </div>
  );
}

function KanbanTab() {
  return (
    <div className="kanban-tab">
      <KanbanBoard />
      <Issues />
    </div>
  );
}

export function App() {
  const health = useResource("health", "/api/health");
  const [tabParam, setTabParam] = useQueryParam("tab", DEFAULT_TAB);
  const tab = asTab(tabParam);
  const [agentsOpen, setAgentsOpen] = useState(false);
  const agentsButton = useRef<HTMLButtonElement>(null);
  const closeAgents = useCallback(() => {
    setAgentsOpen(false);
    agentsButton.current?.focus();
  }, []);
  if (health.unauthorized) return <Unauthorized message={health.error ?? ""} />;
  const selectTab = (next: TabId) => setTabParam(next, "push");
  return (
    <div className="shell">
      <a className="skip-link" href="#main">Skip to status panels</a>
      <DashboardNav ref={agentsButton} tab={tab} onTab={selectTab} agentsOpen={agentsOpen} onToggleAgents={() => (agentsOpen ? closeAgents() : setAgentsOpen(true))} />
      <main id="main" className="main" tabIndex={-1}>
        {tab === "overview" && <Overview />}
        {tab === "kanban" && <KanbanTab />}
        {tab === "runs" && <div className="runs-tab"><RunsTable /></div>}
      </main>
      <TerminalDrawer />
      <RightDrawer open={agentsOpen} onClose={closeAgents} />
    </div>
  );
}
