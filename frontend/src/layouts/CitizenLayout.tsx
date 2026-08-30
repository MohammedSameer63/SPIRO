import { Outlet } from "react-router-dom";
import Sidebar from "../components/Sidebar";

function CitizenLayout() {
  return (
    <div style={styles.container}>
      <Sidebar />

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

export default CitizenLayout;