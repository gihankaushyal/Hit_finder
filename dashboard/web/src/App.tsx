import { PanelFrame } from "./components/PanelFrame";
import { useResource } from "./lib/useResource";
import { Issues } from "./panels/Issues";
import { LatestPr } from "./panels/LatestPr";
import { RightRail } from "./panels/RightRail";
import { TerminalDrawer } from "./panels/TerminalDrawer";
import { TestStatus } from "./panels/TestStatus";
import { TopBar } from "./panels/TopBar";

function KanbanPending() {
  return (
    <PanelFrame title="Kanban" state="ready">
      <p className="dim">Not connected yet.</p>
    </PanelFrame>
  );
}

function Unauthorized({ message }: { message: string }) {
  return (
    <main className="gate" id="main">
      <h1>Open the tokenised URL again</h1>
      <p role="alert">{message}</p>
      <p className="dim">The server prints it on start, as an address ending in <span className="mono">?token=</span> followed by a long code.</p>
    </main>
  );
}

export function App() {
  const health = useResource("health", "/api/health");
  if (health.unauthorized) return <Unauthorized message={health.error ?? ""} />;
  return (
    <div className="shell">
      <a className="skip-link" href="#main">Skip to status panels</a>
      <TopBar />
      <main id="main" className="main" tabIndex={-1}>
        <div className="col col--left"><LatestPr />
          <KanbanPending /></div>
        <div className="col col--middle"><TestStatus />
          <Issues /></div>
        <RightRail />
      </main>
      <TerminalDrawer />
    </div>
  );
}
