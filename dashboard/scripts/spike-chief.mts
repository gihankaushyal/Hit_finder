/** Throwaway M2b spike: in-process MCP tool under safe mode + strictMcpConfig, ask rule reaches canUseTool, resume. */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createSdkMcpServer, query, tool, type CanUseTool } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

const dir = fs.mkdtempSync(path.join(os.homedir(), ".dash-chiefspike-"));
const claude = execFileSync("which", ["claude"], { encoding: "utf8" }).trim();
const env: Record<string, string> = {};
for (const k of ["PATH", "HOME", "USER", "LANG", "TMPDIR", "SHELL"]) if (process.env[k]) env[k] = process.env[k] as string;
const calls: string[] = [];
const server = createSdkMcpServer({ name: "dashboard", version: "1.0.0", tools: [
  tool("ping", "Returns pong with the message. Use it when asked to ping.", { message: z.string() }, async ({ message }) => { calls.push(message); return { content: [{ type: "text", text: `pong:${message}` }] }; }),
] });
const asked: string[] = [];
const canUseTool: CanUseTool = async (name, input) => { asked.push(`${name} ${JSON.stringify(input)}`); return { behavior: "allow", updatedInput: input }; };
async function run(label: string, text: string, resume?: string) {
  async function* input() { yield { type: "user" as const, message: { role: "user" as const, content: text }, parent_tool_use_id: null, session_id: "" }; }
  const q = query({ prompt: input(), options: {
    cwd: dir, model: "haiku", pathToClaudeCodeExecutable: claude, env, settingSources: [], strictMcpConfig: true, extraArgs: { "safe-mode": null }, maxTurns: 4,
    mcpServers: { dashboard: server }, settings: { permissions: { allow: ["Read"], ask: ["mcp__dashboard__ping"], deny: ["Bash", "Edit", "Write"] } }, canUseTool, resume,
  } });
  let sid = ""; let result = ""; let tools: string[] = []; let mcp = "";
  const wd = setTimeout(() => q.close?.(), 90_000);
  for await (const m of q as AsyncIterable<any>) {
    if (m.type === "system" && m.subtype === "init") { sid = m.session_id; tools = (m.tools ?? []).filter((t: string) => t.startsWith("mcp__")); mcp = JSON.stringify(m.mcp_servers); }
    if (m.type === "result") { result = String(m.result).slice(0, 160); q.close?.(); break; }
  }
  clearTimeout(wd);
  console.log(`[${label}] session=${sid.slice(0, 8)} mcp_servers=${mcp} mcp_tools=${JSON.stringify(tools)}\n  result: ${result}`);
  return sid;
}
try {
  const sid = await run("first", "Call the ping tool with the message 'hello', then tell me exactly what it returned.");
  console.log(`  tool handler calls: ${JSON.stringify(calls)}; canUseTool asked: ${JSON.stringify(asked)}`);
  const before = calls.length;
  await run("resume", "Call ping again with the message 'again' and tell me what it returned.", sid);
  console.log(`  handler calls now: ${JSON.stringify(calls)} (new: ${calls.length - before}); asked: ${asked.length}`);
} finally { fs.rmSync(dir, { recursive: true, force: true }); }
