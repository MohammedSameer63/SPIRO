import { apiRequest } from "./client";

export type User = {
  id: string;
  household_id: string | null;
  name: string;
  email: string;
  phone: string | null;
  role: "CITIZEN" | "WORKER" | "ADMIN";
  status: "ACTIVE" | "INACTIVE" | "SUSPENDED";
  created_at: string;
  updated_at: string;
};

export type LoginRequest = {
  email: string;
  password: string;
};

export type AuthResponse = {
  access_token: string;
  token_type: string;
  user: User;
};

export type RegisterRequest = {
  ward_id: string;
  house_number: string;
  street_name: string;
  address: string;
  name: string;
  email: string;
  phone?: string;
  password: string;
};

export async function login(
  data: LoginRequest
): Promise<AuthResponse> {
  return apiRequest<AuthResponse>(
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
): Promise<AuthResponse> {
  return apiRequest<AuthResponse>(
    "/auth/register",
    {
      method: "POST",
      body: JSON.stringify(data),
      auth: false,
    }
  );
}

export async function getCurrentUser(): Promise<User> {
  return apiRequest<User>(
    "/auth/me",
    {
      method: "GET",
    }
  );
}
