import { useEffect, useState } from "react";

import {
  getWorkerStats,
  getWorkerWards,
  getWorkerSchedules,
} from "../../api/reports";

import type {
  WorkerSchedule,
  WorkerWard,
} from "../../api/reports";

function Dashboard() {
  const [stats, setStats] = useState({
    pending: 0,
    accepted: 0,
    in_progress: 0,
    completed_today: 0,
  });

  const [wards, setWards] =
    useState<WorkerWard[]>([]);

  const [schedules, setSchedules] =
    useState<WorkerSchedule[]>([]);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  useEffect(() => {
    loadDashboard();
  }, []);

  async function loadDashboard() {
    try {
      setLoading(true);
      setError("");

      const [
        statsResponse,
        wardsResponse,
        schedulesResponse,
      ] = await Promise.all([
        getWorkerStats(),
        getWorkerWards(),
        getWorkerSchedules(),
      ]);

      if (statsResponse.success) {
        setStats(statsResponse.data);
      }

      if (wardsResponse.success) {
        setWards(wardsResponse.data);
      }

      if (schedulesResponse.success) {
        setSchedules(schedulesResponse.data);
      }
    } catch (err: any) {
      console.error(err);

      setError(
        err?.message ||
          "Failed to load worker dashboard."
      );
    } finally {
      setLoading(false);
    }
  }

  const dashboardStats = [
    {
      title: "Pending Reports",
      value: stats.pending,
    },
    {
      title: "Accepted Reports",
      value: stats.accepted,
    },
    {
      title: "In Progress",
      value: stats.in_progress,
    },
    {
      title: "Completed Today",
      value: stats.completed_today,
    },
  ];

  if (loading) {
    return (
      <div>
        <h1>Worker Dashboard</h1>

        <p style={styles.subtitle}>
          Manage waste reports from your assigned wards.
        </p>

        <p>Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div>
      <h1>Worker Dashboard</h1>

      <p style={styles.subtitle}>
        Manage waste reports from your assigned wards.
      </p>

      {error && (
        <div style={styles.error}>
          {error}

          <button
            onClick={loadDashboard}
            style={styles.retry}
          >
            Retry
          </button>
        </div>
      )}

      {/* Assigned Wards */}

      <div style={styles.wards}>
        <h2>Assigned Wards</h2>

        {wards.length === 0 ? (
          <p style={styles.noWards}>
            No wards assigned to you yet.
          </p>
        ) : (
          <div>
            {wards.map((ward) => (
              <span
                key={ward.id}
                style={styles.ward}
                title={
                  ward.description ||
                  ward.zone ||
                  ""
                }
              >
                {ward.name}
              </span>
            ))}
          </div>
        )}
      </div>

      <div style={styles.schedules}>
        <h2>Collection Schedules</h2>

        {schedules.length === 0 ? (
          <p style={styles.noWards}>
            No collection schedules are available for your assigned wards.
          </p>
        ) : (
          <div style={styles.scheduleList}>
            {schedules.map((schedule) => (
              <div key={schedule.id} style={styles.schedule}>
                <strong>{schedule.ward}</strong>
                <span>{schedule.category}</span>
                <span>
                  {schedule.day}: {schedule.startTime} – {schedule.endTime}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Statistics */}

      <div style={styles.grid}>
        {dashboardStats.map((stat) => (
          <div
            key={stat.title}
            style={styles.card}
          >
            <p style={styles.cardTitle}>
              {stat.title}
            </p>

            <h2 style={styles.value}>
              {stat.value}
            </h2>
          </div>
        ))}
      </div>
    </div>
  );
}

const styles = {
  subtitle: {
    color: "#6b7280",
    marginBottom: "30px",
  },

  wards: {
    marginBottom: "30px",
  },

  schedules: {
    marginBottom: "30px",
  },

  scheduleList: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "10px",
  },

  schedule: {
    display: "flex",
    flexWrap: "wrap" as const,
    gap: "12px",
    alignItems: "center",
    padding: "12px 14px",
    background: "white",
    borderRadius: "8px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
  },

  ward: {
    display: "inline-block",
    marginRight: "10px",
    marginBottom: "8px",
    padding: "8px 14px",
    background: "#dcfce7",
    color: "#166534",
    borderRadius: "20px",
  },

  noWards: {
    color: "#6b7280",
  },

  grid: {
    display: "grid",
    gridTemplateColumns:
      "repeat(auto-fit, minmax(180px, 1fr))",
    gap: "20px",
  },

  card: {
    padding: "25px",
    background: "white",
    borderRadius: "12px",
    boxShadow:
      "0 4px 15px rgba(0,0,0,0.08)",
  },

  cardTitle: {
    color: "#6b7280",
  },

  value: {
    fontSize: "32px",
    margin: "10px 0 0",
  },

  error: {
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

export default Dashboard;
