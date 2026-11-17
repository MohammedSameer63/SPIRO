import { NavLink, useNavigate } from "react-router-dom";
import { LogOut, Recycle } from "lucide-react";
import type { ReactNode } from "react";
import { COLORS, font } from "../theme";

export interface SidebarLink {
  to: string;
  label: string;
  icon: ReactNode;
}

export function AppSidebar({
  title,
  subtitle,
  accent,
  links,
}: {
  title: string;
  subtitle: string;
  accent: string;
  links: SidebarLink[];
}) {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    localStorage.removeItem("role");
    navigate("/login");
  };

  return (
    <aside
      style={{
        width: 248,
        height: "100vh",
        position: "sticky",
        top: 0,
        alignSelf: "flex-start",
        overflowY: "auto",
        padding: "24px 14px",
        background: COLORS.white,
        borderRight: `1px solid ${COLORS.g200}`,
        display: "flex",
        flexDirection: "column",
        fontFamily: font,
        flexShrink: 0,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "0 8px",
        }}
      >
        <div
          style={{
            width: 34,
            height: 34,
            background: accent,
            borderRadius: 9,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#fff",
          }}
        >
          <Recycle size={19} />
        </div>
        <div>
          <div
            style={{
              color: COLORS.g900,
              margin: 0,
              fontSize: 19,
              fontWeight: 900,
              letterSpacing: 0.5,
            }}
          >
            {title}
          </div>
          <div
            style={{
              color: COLORS.g400,
              fontSize: 11,
              marginTop: 1,
            }}
          >
            {subtitle}
          </div>
        </div>
      </div>

      <nav style={{ marginTop: 32, flex: 1 }}>
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            style={({ isActive }: { isActive: boolean }) => ({
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "11px 14px",
              marginBottom: 6,
              borderRadius: 10,
              textDecoration: "none",
              fontSize: 14,
              fontWeight: 600,
              color: isActive ? "white" : COLORS.g700,
              background: isActive ? accent : "transparent",
            })}
          >
            {l.icon}
            {l.label}
          </NavLink>
        ))}
      </nav>

      <button
        onClick={handleLogout}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 8,
          padding: 12,
          border: "none",
          borderRadius: 10,
          background: COLORS.red,
          color: "white",
          fontSize: 14,
          fontWeight: 600,
          cursor: "pointer",
        }}
      >
        <LogOut size={16} /> Logout
      </button>
    </aside>
  );
}
