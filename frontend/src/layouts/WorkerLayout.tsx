import { Outlet } from "react-router-dom";
import {
  CalendarDays,
  ClipboardList,
  LayoutDashboard,
  Route as RouteIcon,
  Truck,
} from "lucide-react";
import WorkerSidebar from "../components/WorkerSidebar";
import { ResponsiveShell } from "../components/ResponsiveShell";
import { COLORS } from "../theme";

function WorkerLayout() {
  return (
    <ResponsiveShell
      sidebar={<WorkerSidebar />}
      title="SPIRO"
      subtitle="Worker Portal"
      accent={COLORS.dBlue}
      tabs={[
        { to: "/worker/dashboard", label: "Home", icon: <LayoutDashboard size={21} /> },
        { to: "/worker/queue", label: "Queue", icon: <ClipboardList size={21} /> },
        { to: "/worker/active", label: "Active", icon: <Truck size={21} /> },
        { to: "/worker/route", label: "Route", icon: <RouteIcon size={21} /> },
        { to: "/worker/schedules", label: "Plan", icon: <CalendarDays size={21} /> },
      ]}
    >
      <Outlet />
    </ResponsiveShell>
  );
}

export default WorkerLayout;
