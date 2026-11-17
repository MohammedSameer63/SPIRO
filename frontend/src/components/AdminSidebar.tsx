import {
  Brain,
  CalendarDays,
  LayoutDashboard,
  ScrollText,
  Users,
} from "lucide-react";
import { AppSidebar } from "./AppSidebar";
import { COLORS } from "../theme";

function AdminSidebar() {
  return (
    <AppSidebar
      title="SPIRO"
      subtitle="Admin Portal"
      accent={COLORS.aPurple}
      links={[
        { to: "/admin/dashboard", label: "Overview", icon: <LayoutDashboard size={17} /> },
        { to: "/admin/audit", label: "Audit Trail", icon: <ScrollText size={17} /> },
        { to: "/admin/workers", label: "Workers", icon: <Users size={17} /> },
        { to: "/admin/schedules", label: "Schedules", icon: <CalendarDays size={17} /> },
        { to: "/admin/ml", label: "AI / ML", icon: <Brain size={17} /> },
      ]}
    />
  );
}

export default AdminSidebar;
