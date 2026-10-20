import { Agents } from "./Agents";
import { Approvals } from "./Approvals";
import { ChiefChat } from "./ChiefChat";

export function RightRail() {
  return (
    <aside className="rail" aria-label="Agents and chief of staff">
      <Approvals />
      <ChiefChat />
      <Agents />
    </aside>
  );
}
