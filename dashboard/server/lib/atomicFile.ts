import fs from "node:fs/promises";
import path from "node:path";

/** `<file>.<pid>.tmp`: what a write leaves behind if the process dies before the rename. */
const TEMP_FILE = /\.(\d+)\.tmp$/;

/** Writes `text` so a reader sees the old file or the new one, never half of it. */
export async function writeAtomic(file: string, text: string): Promise<void> {
  const tmp = `${file}.${process.pid}.tmp`;
  await fs.writeFile(tmp, text);
  await fs.rename(tmp, file);
}

/** Removes temp files that a killed process left in `dir`; this process's own are left alone. */
export async function sweepTemp(dir: string): Promise<void> {
  for (const name of await fs.readdir(dir).catch(() => [] as string[])) {
    const m = TEMP_FILE.exec(name);
    if (m && Number(m[1]) !== process.pid) await fs.rm(path.join(dir, name), { force: true });
  }
}
