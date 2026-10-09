export const UNAUTHORIZED_MESSAGE =
  "The dashboard no longer recognises this browser. Open the tokenised URL printed by the server again.";
const NETWORK_MESSAGE = "Could not reach the dashboard server.";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, credentials: "same-origin" };
  if (method !== "GET") init.headers = { "Content-Type": "application/json" }; // the server requires it on every mutation
  if (body !== undefined) init.body = JSON.stringify(body);
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, NETWORK_MESSAGE);
  }
  if (res.status === 401) throw new ApiError(401, UNAUTHORIZED_MESSAGE);
  if (!res.ok) {
    let message = `Request failed (HTTP ${res.status})`;
    try {
      const data = (await res.json()) as { error?: unknown };
      if (typeof data.error === "string" && data.error !== "") message = data.error;
    } catch {
      // not JSON: keep the status message
    }
    throw new ApiError(res.status, message);
  }
  return (await res.json()) as T;
}

export const apiGet = <T>(path: string): Promise<T> => request<T>("GET", path);
export const apiSend = <T>(method: string, path: string, body?: unknown): Promise<T> => request<T>(method, path, body);
