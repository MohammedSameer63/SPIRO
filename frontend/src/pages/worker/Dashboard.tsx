function Dashboard() {
  const stats = [
    {
      title: "Pending Reports",
      value: 12,
    },
    {
      title: "Accepted Reports",
      value: 4,
    },
    {
      title: "In Progress",
      value: 2,
    },
    {
      title: "Completed Today",
      value: 8,
    },
  ];

  return (
    <div>
      <h1>Worker Dashboard</h1>

      <p style={styles.subtitle}>
        Manage waste reports from your assigned wards.
      </p>

      <div style={styles.wards}>
        <h2>Assigned Wards</h2>

        <span style={styles.ward}>
          Ward 12
        </span>

        <span style={styles.ward}>
          Ward 15
        </span>
      </div>

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

  ward: {
    display: "inline-block",
    marginRight: "10px",
    padding: "8px 14px",
    background: "#dcfce7",
    color: "#166534",
    borderRadius: "20px",
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
};

export default Dashboard;