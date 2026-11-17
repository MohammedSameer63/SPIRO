import { Outlet } from "react-router-dom";
import {
  Brain,
  CalendarDays,
  LayoutDashboard,
  MapPin,
  ScrollText,
  Users,
} from "lucide-react";
import AdminSidebar from "../components/AdminSidebar";
import { ResponsiveShell } from "../components/ResponsiveShell";
import { COLORS } from "../theme";

function AdminLayout() {
  return (
    <ResponsiveShell
      sidebar={<AdminSidebar />}
      title="SPIRO"
      subtitle="Admin Portal"
      accent={COLORS.aPurple}
      tabs={[
        { to: "/admin/dashboard", label: "Home", icon: <LayoutDashboard size={21} /> },
        { to: "/admin/audit", label: "Audit", icon: <ScrollText size={21} /> },
        { to: "/admin/workers", label: "Team", icon: <Users size={21} /> },
        { to: "/admin/wards", label: "Wards", icon: <MapPin size={21} /> },
        { to: "/admin/schedules", label: "Plan", icon: <CalendarDays size={21} /> },
        { to: "/admin/ml", label: "AI", icon: <Brain size={21} /> },
      ]}
    >
      <Outlet />
    </ResponsiveShell>
  );
}

export default AdminLayout;
