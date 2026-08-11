function Dashboard() {
  const stats = [
    {
      title: "Total Citizens",
      value: 520,
    },
    {
      title: "Total Workers",
      value: 18,
    },
    {
      title: "Total Reports",
      value: 340,
    },
    {
      title: "Pending Reports",
      value: 39,
    },
    {
      title: "Completed Reports",
      value: 301,
    },
  ];

  const wards = [
    {
      ward: "Ward 12",
      reports: 52,
      pending: 4,
    },
    {
      ward: "Ward 15",
      reports: 41,
      pending: 7,
    },
    {
      ward: "Ward 18",
      reports: 36,
      pending: 3,
    },
  ];

  const workers = [
    {
      name: "John",
      activeReports: 6,
      completedToday: 8,
    },
    {
      name: "Rahul",
      activeReports: 4,
      completedToday: 6,
    },
    {
      name: "Arun",
      activeReports: 2,
      completedToday: 5,
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

              {wards.map((ward) => (
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
              ))}

            </tbody>

          </table>

        </div>

      </section>

      {/* Worker Overview */}

      <section style={styles.section}>

        <h2>
          Worker Overview
        </h2>

        <div style={styles.tableWrapper}>

          <table style={styles.table}>

            <thead>
              <tr>

                <th style={styles.th}>
                  Worker
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

              {workers.map((worker) => (
                <tr key={worker.name}>

                  <td style={styles.td}>
                    {worker.name}
                  </td>

                  <td style={styles.td}>
                    {worker.activeReports}
                  </td>

                  <td style={styles.td}>
                    {worker.completedToday}
                  </td>

                </tr>
              ))}

            </tbody>

          </table>

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
    borderCollapse: "collapse" as const,
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
};

export default Dashboard;