import { Outlet } from "react-router-dom";
import WorkerSidebar from "../components/WorkerSidebar";

function WorkerLayout() {
  return (
    <div style={styles.container}>
      <WorkerSidebar />

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

export default WorkerLayout;