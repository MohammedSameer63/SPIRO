import { Outlet } from "react-router-dom";
import AdminSidebar from "../components/AdminSidebar";

function AdminLayout() {
  return (
    <div style={styles.container}>

      <AdminSidebar />

      <main style={styles.main}>
        <Outlet />
      </main>

    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    display: "flex",
    background: "#f4f7f5",
  },

  main: {
    flex: 1,
    padding: "40px",
  },
};

export default AdminLayout;