import { useEffect, useState } from "react";
import { getWorkerSchedules } from "../../api/reports";
import type { WorkerSchedule } from "../../api/reports";

function Schedules() {
  const [schedules, setSchedules] = useState<
    WorkerSchedule[]
  >([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadSchedules();
  }, []);

  async function loadSchedules() {
    try {
      setLoading(true);
      setError("");

      const response =
        await getWorkerSchedules();

      if (response.success) {
        setSchedules(response.data);
      }
    } catch (err: any) {
      console.error(err);

      setError(
        err?.message ||
          "Failed to load collection schedules."
      );
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div>
        <h1>Collection Schedules</h1>

        <p style={styles.subtitle}>
          View collection schedules for your
          assigned wards.
        </p>

        <div style={styles.loading}>
          Loading schedules...
        </div>
      </div>
    );
  }

  return (
    <div>
      <div style={styles.header}>
        <div>
          <h1>
            Collection Schedules
          </h1>

          <p style={styles.subtitle}>
            View collection schedules for your
            assigned wards.
          </p>
        </div>

        <button
          onClick={loadSchedules}
          style={styles.refresh}
        >
          Refresh
        </button>
      </div>

      {error && (
        <div style={styles.error}>
          <span>{error}</span>

          <button
            onClick={loadSchedules}
            style={styles.retry}
          >
            Retry
          </button>
        </div>
      )}

      {schedules.length === 0 ? (
        <div style={styles.empty}>
          <h2>
            No schedules assigned
          </h2>

          <p>
            There are currently no collection
            schedules for your assigned wards.
          </p>
        </div>
      ) : (
        <div style={styles.grid}>
          {schedules.map((schedule) => (
            <div
              key={schedule.id}
              style={styles.card}
            >
              <div style={styles.cardHeader}>
                <h2 style={styles.ward}>
                  {schedule.ward}
                </h2>

                <span style={styles.category}>
                  {schedule.category}
                </span>
              </div>

              <div style={styles.info}>
                <div style={styles.row}>
                  <strong>
                    Day
                  </strong>

                  <span>
                    {schedule.day}
                  </span>
                </div>

                <div style={styles.row}>
                  <strong>
                    Collection Time
                  </strong>

                  <span>
                    {schedule.startTime}
                    {" - "}
                    {schedule.endTime}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "25px",
  },

  subtitle: {
    color: "#6b7280",
  },

  refresh: {
    padding: "10px 16px",
    border: "none",
    borderRadius: "7px",
    background: "#15803d",
    color: "white",
    cursor: "pointer",
  },

  loading: {
    padding: "30px",
    background: "white",
    borderRadius: "10px",
    color: "#6b7280",
  },

  grid: {
    display: "grid",
    gridTemplateColumns:
      "repeat(auto-fit, minmax(280px, 1fr))",
    gap: "20px",
  },

  card: {
    padding: "22px",
    background: "white",
    borderRadius: "12px",
    boxShadow:
      "0 4px 15px rgba(0,0,0,0.08)",
  },

  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "15px",
    marginBottom: "20px",
  },

  ward: {
    margin: 0,
  },

  category: {
    padding: "6px 12px",
    borderRadius: "20px",
    background: "#dcfce7",
    color: "#166534",
    fontSize: "13px",
    fontWeight: "bold",
  },

  info: {
    borderTop:
      "1px solid #e5e7eb",
    paddingTop: "15px",
  },

  row: {
    display: "flex",
    justifyContent: "space-between",
    gap: "20px",
    padding: "10px 0",
  },

  empty: {
    padding: "40px",
    background: "white",
    borderRadius: "12px",
    textAlign: "center" as const,
    color: "#6b7280",
  },

  error: {
    display: "flex",
    alignItems: "center",
    padding: "15px",
    marginBottom: "20px",
    background: "#fee2e2",
    color: "#b91c1c",
    borderRadius: "8px",
  },

  retry: {
    marginLeft: "15px",
    padding: "8px 14px",
    border: "none",
    borderRadius: "6px",
    background: "#15803d",
    color: "white",
    cursor: "pointer",
  },
};

export default Schedules;