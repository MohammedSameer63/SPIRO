import { API_BASE_URL } from "./config";

type RequestOptions = RequestInit & {
  auth?: boolean;
};

export async function apiRequest<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const {
    auth = true,
    headers,
    ...fetchOptions
  } = options;

  const token = localStorage.getItem("token");

  const requestHeaders = new Headers(
    headers
  );

  if (
    fetchOptions.body &&
    !(fetchOptions.body instanceof FormData)
  ) {
    requestHeaders.set(
      "Content-Type",
      "application/json"
    );
  }

  if (auth && token) {
    requestHeaders.set(
      "Authorization",
      `Bearer ${token}`
    );
  }

  const response = await fetch(
    `${API_BASE_URL}${endpoint}`,
    {
      ...fetchOptions,
      headers: requestHeaders,
    }
  );

  let result: any = null;

  try {
    result = await response.json();
  } catch {
    result = null;
  }

  if (!response.ok) {
    throw new Error(
      result?.message ||
        `Request failed with status ${response.status}`
    );
  }

  return result as T;
}