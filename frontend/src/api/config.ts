// Base URL of the FastAPI backend (without trailing slash).
//
// Resolution order (so demos survive Wi-Fi/hotspot switches with zero
// reconfiguration):
//   1. VITE_API_BASE_URL env var, when set (production builds).
//   2. Otherwise, the page's own hostname: if the app was opened via a
//      LAN IP (phone demo), the backend lives on the same host.
//   3. Otherwise localhost default (laptop dev).
function isIp(host: string): boolean {
  return /^[0-9.]+$/.test(host) || host.includes(":");
}

function defaultBaseUrl(): string {
  if (
    typeof window !== "undefined" &&
    window.location?.hostname
  ) {
    const host = window.location.hostname;
    if (isIp(host)) {
      return `http://${host}:8000/api/v1`;
    }
    if (
      host === "localhost" ||
      host === "127.0.0.1"
    ) {
      return "http://127.0.0.1:8000/api/v1";
    }
    // Hosted domain without an explicit env var: assume same-origin API.
    const scheme =
      window.location.protocol === "http:" ? "http:" : "https:";
    return `${scheme}//${host}/api/v1`;
  }
  return "http://127.0.0.1:8000/api/v1";
}

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL as string | undefined
)?.replace(/\/+$/, "") || defaultBaseUrl();

// Origin of the backend (scheme + host), used to resolve relative
// image paths like "/uploads/reports/abc.jpg" served by FastAPI.
export const API_ORIGIN = API_BASE_URL.replace(
  /\/api\/v\d+$/,
  ""
);

export function resolveImageUrl(
  imageUrl: string | null | undefined
): string {
  if (!imageUrl) return "";
  if (
    imageUrl.startsWith("http://") ||
    imageUrl.startsWith("https://")
  ) {
    return imageUrl;
  }
  return `${API_ORIGIN}${
    imageUrl.startsWith("/") ? "" : "/"
  }${imageUrl}`;
}
