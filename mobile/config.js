import axios from 'axios';

function resolveApiBaseUrl() {
  if (process.env.EXPO_PUBLIC_API_URL) {
    return process.env.EXPO_PUBLIC_API_URL;
  }

  return 'https://hemolens.onrender.com';
}

export const API_BASE_URL = resolveApiBaseUrl();

// Render free services spin down when idle.  The first request can take up to a
// minute while the container starts, so a short probe makes Expo Go report the
// API as offline even though it is still waking up.
const BACKEND_PROBE_TIMEOUT_MS = 70000;
const BACKEND_PROBE_RETRY_DELAYS_MS = [0, 3000];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function probeBackend() {
  let lastError = null;

  for (const delay of BACKEND_PROBE_RETRY_DELAYS_MS) {
    if (delay > 0) {
      await sleep(delay);
    }

    try {
      const response = await axios.get(`${API_BASE_URL}/health`, {
        timeout: BACKEND_PROBE_TIMEOUT_MS,
      });

      return { ok: true, healthData: response.data || {} };
    } catch (error) {
      lastError = error;
    }
  }

  return { ok: false, error: lastError };
}
