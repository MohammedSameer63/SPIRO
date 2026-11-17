import { useEffect, useState } from "react";
import {
  Navigate,
  Outlet,
  useLocation,
} from "react-router-dom";
import {
  getCurrentUser,
  type User,
} from "../api/auth";

type Role = User["role"];

const ROLE_HOME: Record<Role, string> = {
  CITIZEN: "/citizen/dashboard",
  WORKER: "/worker/dashboard",
  ADMIN: "/admin/dashboard",
};

interface RequireRoleProps {
  roles: Role[];
}

/*
 * Route guard for role sections (/citizen, /worker, /admin).
 * Without this, opening a section with another role's login just
 * 403s on every endpoint. Instead we bounce the user to the right
 * place: login when unauthenticated, their own dashboard on mismatch.
 */
export function RequireRole({
  roles,
}: RequireRoleProps) {
  const location = useLocation();

  const [user, setUser] =
    useState<User | null>(null);
  const [loading, setLoading] =
    useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const current = await getCurrentUser();
        if (!cancelled) setUser(current);
      } catch {
        if (!cancelled) setUser(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    if (!localStorage.getItem("token")) {
      setLoading(false);
      return;
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [location.pathname]);

  if (loading) {
    return (
      <div style={{ padding: "40px" }}>
        <p>Loading...</p>
      </div>
    );
  }

  if (!user) {
    return (
      <Navigate to="/login" replace />
    );
  }

  if (!roles.includes(user.role)) {
    return (
      <Navigate
        to={ROLE_HOME[user.role]}
        replace
      />
    );
  }

  return <Outlet />;
}
