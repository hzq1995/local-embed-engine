const defaultApiBase = "/api";

export function getErrorMessage(error) {
  if (error instanceof Error) return error.message;
  return String(error);
}

export async function apiRequest(path, payload) {
  const response = await fetch(`${defaultApiBase}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  let data = null;
  try {
    data = await response.json();
  } catch (_error) {
    data = null;
  }

  if (!response.ok) {
    const detail = data?.detail;
    if (typeof detail === "string") throw new Error(detail);
    if (detail) throw new Error(JSON.stringify(detail));
    throw new Error(`${response.status} ${response.statusText}`);
  }

  return data;
}

export async function checkHealth() {
  const response = await fetch(`${defaultApiBase}/health`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return await response.json();
}
