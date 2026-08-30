import { useEffect, useState } from "react";
import { getAdminDashboard } from "../../api/admin";

function Dashboard() {
  const [data, setData] =
    useState<{
      stats: {
        total_citizens: number;
        total_workers: number;
        total_reports: number;
        pending_reports: number;
        completed_reports: number;
      };
      workers: {
        id: string;
        name: string;
        email: string;
        active_reports: number;
        completed_today: number;
      }[];
      wards: {
        ward: string;
        reports: number;
        pending: number;
      }[];
    } | null>(null);

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

      const response =
        await getAdminDashboard();

      if (response.success) {
        setData(response.data);
      }
    } catch (err: any) {
      console.error(err);

      setError(
        err?.message ||
          "Failed to load admin dashboard."
      );
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div>
        <h1>Admin Dashboard</h1>

        <p style={styles.subtitle}>
          Municipality-wide waste management overview.
        </p>

        <p>Loading dashboard...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <h1>Admin Dashboard</h1>

        <div style={styles.error}>
          {error}

          <button
            onClick={loadDashboard}
            style={styles.retry}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const stats = [
    {
      title: "Total Citizens",
      value: data.stats.total_citizens,
    },
    {
      title: "Total Workers",
      value: data.stats.total_workers,
    },
    {
      title: "Total Reports",
      value: data.stats.total_reports,
    },
    {
      title: "Pending Reports",
      value: data.stats.pending_reports,
    },
    {
      title: "Completed Reports",
      value: data.stats.completed_reports,
    },
  ];

  return (
    <div>
      <h1>
        Admin Dashboard
      </h1>

      <p style={styles.subtitle}>
        Municipality-wide waste management overview.
      </p>

      {/* Statistics */}

      <div style={styles.grid}>
        {stats.map((stat) => (
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

      {/* Ward Overview */}

      <section style={styles.section}>
        <h2>
          Ward Overview
        </h2>

        <div style={styles.tableWrapper}>
          {data.wards.length === 0 ? (
            <div style={styles.empty}>
              Ward data is not available yet.
              <br />
              Ward assignment has not been
              connected to the database.
            </div>
          ) : (
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>
                    Ward
                  </th>

                  <th style={styles.th}>
                    Reports
                  </th>

                  <th style={styles.th}>
                    Pending
                  </th>
                </tr>
              </thead>

              <tbody>
                {data.wards.map(
                  (ward) => (
                    <tr key={ward.ward}>
                      <td style={styles.td}>
                        {ward.ward}
                      </td>

                      <td style={styles.td}>
                        {ward.reports}
                      </td>

                      <td style={styles.td}>
                        {ward.pending}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          )}
        </div>
      </section>

      {/* Worker Overview */}

      <section style={styles.section}>
        <h2>
          Worker Overview
        </h2>

        <div style={styles.tableWrapper}>
          {data.workers.length === 0 ? (
            <div style={styles.empty}>
              No active workers found.
            </div>
          ) : (
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>
                    Worker
                  </th>

                  <th style={styles.th}>
                    Email
                  </th>

                  <th style={styles.th}>
                    Active Reports
                  </th>

                  <th style={styles.th}>
                    Completed Today
                  </th>
                </tr>
              </thead>

              <tbody>
                {data.workers.map(
                  (worker) => (
                    <tr key={worker.id}>
                      <td style={styles.td}>
                        {worker.name}
                      </td>

                      <td style={styles.td}>
                        {worker.email}
                      </td>

                      <td style={styles.td}>
                        {worker.active_reports}
                      </td>

                      <td style={styles.td}>
                        {worker.completed_today}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </div>
  );
}

const styles = {
  subtitle: {
    color: "#6b7280",
    marginBottom: "30px",
  },

  grid: {
    display: "grid",
    gridTemplateColumns:
      "repeat(auto-fit, minmax(180px, 1fr))",
    gap: "20px",
  },

  card: {
    background: "white",
    padding: "25px",
    borderRadius: "12px",
    boxShadow:
      "0 4px 15px rgba(0,0,0,0.08)",
  },

  cardTitle: {
    color: "#6b7280",
    margin: 0,
  },

  value: {
    fontSize: "30px",
    marginTop: "10px",
  },

  section: {
    marginTop: "35px",
  },

  tableWrapper: {
    background: "white",
    borderRadius: "12px",
    overflow: "hidden" as const,
    boxShadow:
      "0 4px 15px rgba(0,0,0,0.08)",
  },

  table: {
    width: "100%",
    borderCollapse:
      "collapse" as const,
  },

  th: {
    padding: "15px",
    textAlign: "left" as const,
    background: "#f0fdf4",
    borderBottom:
      "1px solid #e5e7eb",
  },

  td: {
    padding: "15px",
    borderBottom:
      "1px solid #e5e7eb",
  },

  empty: {
    padding: "30px",
    textAlign: "center" as const,
    color: "#6b7280",
  },

  error: {
    padding: "15px",
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