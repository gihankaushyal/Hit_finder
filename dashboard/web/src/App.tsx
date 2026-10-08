import { PanelFrame } from "./components/PanelFrame";
import { useResource } from "./lib/useResource";
import { RightRail } from "./panels/RightRail";
import { TerminalDrawer } from "./panels/TerminalDrawer";
import { TopBar } from "./panels/TopBar";

const PENDING_PANELS_LEFT = ["Latest pull request", "Kanban"];
const PENDING_PANELS_MIDDLE = ["Tests", "Issues"];

function Pending({ title }: { title: string }) {
  return (
    <PanelFrame title={title} state="ready">
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
        <div className="col col--left">{PENDING_PANELS_LEFT.map((t) => <Pending key={t} title={t} />)}</div>
        <div className="col col--middle">{PENDING_PANELS_MIDDLE.map((t) => <Pending key={t} title={t} />)}</div>
        <RightRail />
      </main>
      <TerminalDrawer />
    </div>
  );
}
