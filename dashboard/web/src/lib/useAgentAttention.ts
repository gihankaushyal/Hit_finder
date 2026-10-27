import { useEffect, useMemo, useReducer, useRef } from "react";
import type { AgentsResponse, ApprovalsResponse, ChiefSummary } from "../../../shared/types";
import { EMPTY_ATTENTION, deriveBubble, isUnseenTransition, reduceAttention, unseenCount, type Bubble } from "./attention";
import { notifyFinished, outcomeText, pushToast, setTabAttention } from "./attentionEffects";
import { subscribe } from "./events";
import { AGENTS_REFRESH_MS, useResource } from "./useResource";

/** Matches CHIEF_ID in server/agents/chief.ts. */
const CHIEF_ID = "chief";
const RUNNING = ["starting", "running"];

/**
 * The state of the Agents button's bubble: red for failures and approvals, yellow while agents run
 * (shelf closed), green for finished results you have not looked at. Also drives the toast, the browser
 * notification, the tab title and the favicon.
 */
export function useAgentAttention(agentsOpen: boolean): Bubble {
  const [state, dispatch] = useReducer(reduceAttention, EMPTY_ATTENTION);
  const agents = useResource<AgentsResponse>("agents", "/api/agents", (d) => d.agents.length === 0, AGENTS_REFRESH_MS);
  const chief = useResource<ChiefSummary>("chat", "/api/chief", () => false, AGENTS_REFRESH_MS);
  const approvals = useResource<ApprovalsResponse>("approvals", "/api/approvals", (d) => d.approvals.length === 0, AGENTS_REFRESH_MS);
  const open = useRef(agentsOpen);
  open.current = agentsOpen;

  useEffect(() => {
    const running = (agents.data?.agents ?? []).filter((a) => RUNNING.includes(a.state)).map((a) => a.id);
    if (chief.data && RUNNING.includes(chief.data.state)) running.push(CHIEF_ID);
    dispatch({ kind: "seed", running });
  }, [agents.data, chief.data]);

  useEffect(
    () =>
      subscribe({
        resource: "agents",
        reload: () => undefined, // useResource already refetches on reconnect; the effect above re-seeds
        onAgentState: (ev) => {
          dispatch({ kind: "event", id: ev.id, state: ev.state, open: open.current });
          if (!isUnseenTransition(ev.state, open.current)) return;
          pushToast(outcomeText(ev.name, ev.state), ev.state === "failed" ? "red" : "green");
          notifyFinished(ev.name, ev.state);
        },
      }),
    [],
  );

  useEffect(() => {
    if (agentsOpen) dispatch({ kind: "open" });
  }, [agentsOpen]);

  const waiting = approvals.data?.approvals.length ?? 0;
  const bubble = useMemo(() => deriveBubble(state, waiting, agentsOpen), [state, waiting, agentsOpen]);
  const unseen = unseenCount(state);
  useEffect(() => {
    setTabAttention(unseen, bubble.color);
  }, [unseen, bubble.color]);
  return bubble;
}
