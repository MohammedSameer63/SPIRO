import { apiRequest } from "./client";

export type LoginRequest = {
  email: string;
  password: string;
};

export type LoginResponse = {
  success: boolean;
  message: string;
  data: {
    token: string;
    user: {
      id: string;
      name: string;
      role: "CITIZEN" | "WORKER" | "ADMIN";
    };
  };
};

export type RegisterRequest = {
  name: string;
  email: string;
  password: string;
  phone?: string;
  wardId: string;
  houseNumber: string;
  streetName?: string;
  address: string;
};

export async function login(
  data: LoginRequest
) {
  return apiRequest<LoginResponse>(
    "/auth/login",
    {
      method: "POST",
      body: JSON.stringify(data),
      auth: false,
    }
  );
}

export async function register(
  data: RegisterRequest
) {
  return apiRequest(
    "/auth/register",
    {
      method: "POST",
      body: JSON.stringify(data),
      auth: false,
    }
  );
}

export async function getCurrentUser() {
  return apiRequest(
    "/auth/me",
    {
      method: "GET",
    }
  );
}

export async function logout() {
  return apiRequest(
    "/auth/logout",
    {
      method: "POST",
    }
  );
}