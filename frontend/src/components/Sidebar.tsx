import {
  Bell,
  ClipboardList,
  Home,
  PlusSquare,
  QrCode,
  Wallet,
} from "lucide-react";
import { AppSidebar } from "./AppSidebar";
import { COLORS } from "../theme";

function Sidebar() {
  return (
    <AppSidebar
      title="SPIRO"
      subtitle="Citizen Portal"
      accent={COLORS.cGreen}
      links={[
        { to: "/citizen/dashboard", label: "Dashboard", icon: <Home size={17} /> },
        { to: "/citizen/report", label: "Submit Report", icon: <PlusSquare size={17} /> },
        { to: "/citizen/reports", label: "My Reports", icon: <ClipboardList size={17} /> },
        { to: "/citizen/wallet", label: "Wallet", icon: <Wallet size={17} /> },
        { to: "/citizen/qr", label: "My QR", icon: <QrCode size={17} /> },
        { to: "/citizen/alerts", label: "Alerts", icon: <Bell size={17} /> },
      ]}
    />
  );
}

export default Sidebar;
