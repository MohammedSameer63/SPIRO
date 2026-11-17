import { NavLink, useNavigate } from "react-router-dom";
import { LogOut, Recycle } from "lucide-react";
import type { ReactNode } from "react";
import { useIsMobile } from "../hooks/useMediaQuery";
import { COLORS, font } from "../theme";
import type { SidebarLink } from "./AppSidebar";

/*
 * Desktop: classic sidebar + centered content column.
 * Mobile (<=860px): compact top header + full-width content +
 * fixed bottom tab bar (thumb-friendly, the mobile pattern).
 */
export function ResponsiveShell({
  sidebar,
  title,
  subtitle,
  accent,
  tabs,
  children,
}: {
  sidebar: ReactNode;
  title: string;
  subtitle: string;
  accent: string;
  tabs: SidebarLink[];
  children: ReactNode;
}) {
  const isMobile = useIsMobile();
  const navigate = useNavigate();

  if (!isMobile) {
    return (
      <div style={styles.desktopContainer}>
        {sidebar}
        <main style={styles.desktopMain}>{children}</main>
      </div>
    );
  }

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    localStorage.removeItem("role");
    navigate("/login");
  };

  return (
    <div style={styles.mobileContainer}>
      <header style={styles.topBar}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <div
            style={{
              width: 30,
              height: 30,
              background: accent,
              borderRadius: 8,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
            }}
          >
            <Recycle size={17} />
          </div>
          <div>
            <div style={styles.topTitle}>{title}</div>
            <div style={styles.topSubtitle}>{subtitle}</div>
          </div>
        </div>
        <button
          onClick={handleLogout}
          aria-label="Logout"
          style={{ ...styles.iconBtn, color: COLORS.g500 }}
        >
          <LogOut size={19} />
        </button>
      </header>

      <main style={styles.mobileMain}>{children}</main>

      <nav style={styles.tabBar}>
        {tabs.map((t) => (
          <NavLink
            key={t.to}
            to={t.to}
            style={({ isActive }: { isActive: boolean }) => ({
              flex: "1 1 0",
              minWidth: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 2,
              padding: "8px 2px 10px",
              textDecoration: "none",
              color: isActive ? accent : COLORS.g400,
            })}
          >
            {({ isActive }: { isActive: boolean }) => (
              <>
                <span
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 44,
                    height: 28,
                    borderRadius: 999,
                    background: isActive ? accent + "18" : "transparent",
                  }}
                >
                  {t.icon}
                </span>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    maxWidth: "100%",
                  }}
                >
                  {t.label}
                </span>
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

const styles = {
  desktopContainer: {
    minHeight: "100vh",
    display: "flex",
    background: "#f4f7f5",
    fontFamily: font,
  } as React.CSSProperties,

  desktopMain: {
    flex: 1,
    minWidth: 0,
    padding: "32px 40px",
    maxWidth: 1240,
    width: "100%",
    margin: "0 auto",
  } as React.CSSProperties,

  mobileContainer: {
    minHeight: "100dvh",
    display: "flex",
    flexDirection: "column",
    background: "#f4f7f5",
    fontFamily: font,
  } as React.CSSProperties,

  topBar: {
    position: "sticky",
    top: 0,
    zIndex: 50,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "10px 14px",
    background: COLORS.white,
    borderBottom: `1px solid ${COLORS.g200}`,
  } as React.CSSProperties,

  topTitle: {
    fontSize: 16,
    fontWeight: 900,
    color: COLORS.g900,
    lineHeight: 1.1,
  } as React.CSSProperties,

  topSubtitle: {
    fontSize: 10,
    color: COLORS.g400,
  } as React.CSSProperties,

  iconBtn: {
    border: "none",
    background: COLORS.g100,
    width: 38,
    height: 38,
    borderRadius: 10,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
  } as React.CSSProperties,

  mobileMain: {
    flex: 1,
    minWidth: 0,
    padding: "16px 14px 96px",
    width: "100%",
    maxWidth: 640,
    margin: "0 auto",
  } as React.CSSProperties,

  tabBar: {
    position: "fixed",
    bottom: 0,
    left: 0,
    right: 0,
    zIndex: 50,
    display: "flex",
    background: COLORS.white,
    borderTop: `1px solid ${COLORS.g200}`,
    paddingBottom: "env(safe-area-inset-bottom, 0px)",
    boxShadow: "0 -2px 12px rgba(0,0,0,0.06)",
  } as React.CSSProperties,
};
