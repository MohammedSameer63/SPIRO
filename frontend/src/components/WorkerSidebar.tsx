import {
  CalendarDays,
  ClipboardList,
  LayoutDashboard,
  Route as RouteIcon,
  Truck,
} from "lucide-react";
import { AppSidebar } from "./AppSidebar";
import { COLORS } from "../theme";

function WorkerSidebar() {
  return (
    <AppSidebar
      title="SPIRO"
      subtitle="Worker Portal"
      accent={COLORS.dBlue}
      links={[
        { to: "/worker/dashboard", label: "Dashboard", icon: <LayoutDashboard size={17} /> },
        { to: "/worker/queue", label: "Report Queue", icon: <ClipboardList size={17} /> },
        { to: "/worker/active", label: "Active Reports", icon: <Truck size={17} /> },
        { to: "/worker/route", label: "Today's Route", icon: <RouteIcon size={17} /> },
        { to: "/worker/schedules", label: "Schedules", icon: <CalendarDays size={17} /> },
      ]}
    />
  );
}

export default WorkerSidebar;
