import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const SAFE_SESSION_ID = /^[A-Za-z0-9-]{1,128}$/;

/** True unless the saved SDK transcript for a session is positively gone. */
export type TranscriptProbe = (sessionId: string) => Promise<boolean>;

/**
 * Looks for `<configDir>/projects/<any project>/<sessionId>.jsonl`, which is where the SDK keeps a session's
 * transcript (30 days by default). It scans every project folder rather than computing the folder name, so it
 * does not depend on how the SDK encodes the working directory. When it cannot tell (unreadable folder, odd id)
 * it answers true: a wrong "gone" would lock a good conversation, while a wrong "there" is caught when the
 * session fails to resume.
 */
export function createTranscriptProbe(configDir: string = process.env.CLAUDE_CONFIG_DIR ?? path.join(os.homedir(), ".claude")): TranscriptProbe {
  const projects = path.join(configDir, "projects");
  return async (sessionId) => {
    if (!SAFE_SESSION_ID.test(sessionId)) return true;
    let folders: string[];
    try {
      folders = await fs.readdir(projects);
    } catch {
      return true;
    }
    for (const folder of folders) {
      try {
        await fs.access(path.join(projects, folder, `${sessionId}.jsonl`));
        return true;
      } catch {
        /* not in this folder */
      }
    }
    return false;
  };
}
