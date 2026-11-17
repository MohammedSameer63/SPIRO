import { Outlet } from "react-router-dom";
import {
  Bell,
  ClipboardList,
  Home,
  PlusSquare,
  QrCode,
  Wallet,
} from "lucide-react";
import Sidebar from "../components/Sidebar";
import { ResponsiveShell } from "../components/ResponsiveShell";
import { COLORS } from "../theme";

function CitizenLayout() {
  return (
    <ResponsiveShell
      sidebar={<Sidebar />}
      title="SPIRO"
      subtitle="Citizen Portal"
      accent={COLORS.cGreen}
      tabs={[
        { to: "/citizen/dashboard", label: "Home", icon: <Home size={21} /> },
        { to: "/citizen/report", label: "Submit", icon: <PlusSquare size={21} /> },
        { to: "/citizen/reports", label: "Reports", icon: <ClipboardList size={21} /> },
        { to: "/citizen/wallet", label: "Wallet", icon: <Wallet size={21} /> },
        { to: "/citizen/qr", label: "My QR", icon: <QrCode size={21} /> },
        { to: "/citizen/alerts", label: "Alerts", icon: <Bell size={21} /> },
      ]}
    >
      <Outlet />
    </ResponsiveShell>
  );
}

export default CitizenLayout;
