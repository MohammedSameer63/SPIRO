import { useState } from "react";

const initialWorkers = [
  {
    workerId: "worker-001",
    name: "John",
    wards: ["Ward 12", "Ward 15"],
  },
  {
    workerId: "worker-002",
    name: "Rahul",
    wards: ["Ward 18"],
  },
  {
    workerId: "worker-003",
    name: "Arun",
    wards: ["Ward 12"],
  },
];

const availableWards = [
  "Ward 12",
  "Ward 15",
  "Ward 18",
  "Ward 20",
];

function Workers() {
  const [workers, setWorkers] =
    useState(initialWorkers);

  const [selectedWorker, setSelectedWorker] =
    useState<string | null>(null);

  const [selectedWard, setSelectedWard] =
    useState("");

  const handleAssign = () => {
    if (!selectedWorker) {
      return;
    }

    if (!selectedWard) {
      alert("Please select a ward.");
      return;
    }

    setWorkers((currentWorkers) =>
      currentWorkers.map((worker) => {
        if (
          worker.workerId !== selectedWorker
        ) {
          return worker;
        }

        if (
          worker.wards.includes(selectedWard)
        ) {
          alert(
            "Worker is already assigned to this ward."
          );

          return worker;
        }

        if (worker.wards.length >= 5) {
          alert(
            "A worker can be assigned to a maximum of 5 wards."
          );

          return worker;
        }

        return {
          ...worker,
          wards: [
            ...worker.wards,
            selectedWard,
          ],
        };
      })
    );

    setSelectedWorker(null);
    setSelectedWard("");
  };

  return (
    <div>
      <h1>Worker Management</h1>

      <p style={styles.subtitle}>
        Manage workers and their assigned wards.
      </p>

      <div style={styles.container}>
        {workers.map((worker) => (
          <div
            key={worker.workerId}
            style={styles.card}
          >
            <div style={styles.header}>
              <div>
                <h2>{worker.name}</h2>

                <p style={styles.id}>
                  ID: {worker.workerId}
                </p>
              </div>

              <button
                onClick={() => {
                  setSelectedWorker(
                    worker.workerId
                  );
                  setSelectedWard("");
                }}
                style={styles.button}
              >
                + Assign Ward
              </button>
            </div>

            <h3>Assigned Wards</h3>

            <div>
              {worker.wards.map((ward) => (
                <span
                  key={ward}
                  style={styles.ward}
                >
                  {ward}
                </span>
              ))}
            </div>

            <p style={styles.count}>
              {worker.wards.length}/5 wards assigned
            </p>

            {/* Assignment Form */}

            {selectedWorker ===
              worker.workerId && (
              <div style={styles.form}>
                <label>
                  Select Ward
                </label>

                <select
                  value={selectedWard}
                  onChange={(e) =>
                    setSelectedWard(
                      e.target.value
                    )
                  }
                  style={styles.select}
                >
                  <option value="">
                    -- Select Ward --
                  </option>

                  {availableWards.map(
                    (ward) => (
                      <option
                        key={ward}
                        value={ward}
                      >
                        {ward}
                      </option>
                    )
                  )}
                </select>

                <div>
                  <button
                    onClick={handleAssign}
                    style={styles.confirm}
                  >
                    Assign
                  </button>

                  <button
                    onClick={() => {
                      setSelectedWorker(null);
                      setSelectedWard("");
                    }}
                    style={styles.cancel}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

const styles = {
  subtitle: {
    color: "#6b7280",
    marginBottom: "25px",
  },

  container: {
    maxWidth: "900px",
    display: "flex",
    flexDirection: "column" as const,
    gap: "20px",
  },

  card: {
    background: "white",
    padding: "25px",
    borderRadius: "12px",
    boxShadow:
      "0 4px 15px rgba(0,0,0,0.08)",
  },

  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },

  id: {
    color: "#9ca3af",
    fontSize: "13px",
  },

  button: {
    padding: "10px 15px",
    border: "none",
    borderRadius: "7px",
    background: "#15803d",
    color: "white",
    cursor: "pointer",
    fontSize: "14px",
  },

  ward: {
    display: "inline-block",
    padding: "8px 12px",
    marginRight: "8px",
    marginBottom: "8px",
    background: "#dcfce7",
    color: "#166534",
    borderRadius: "20px",
  },

  count: {
    color: "#6b7280",
    fontSize: "13px",
  },

  form: {
    marginTop: "20px",
    padding: "20px",
    background: "#f0fdf4",
    borderRadius: "10px",
    border: "1px solid #bbf7d0",
  },

  select: {
    display: "block",
    width: "100%",
    maxWidth: "400px",
    padding: "10px",
    marginTop: "8px",
    marginBottom: "15px",
    borderRadius: "6px",
    border: "1px solid #d1d5db",
  },

  confirm: {
    padding: "9px 18px",
    border: "none",
    borderRadius: "6px",
    background: "#15803d",
    color: "white",
    cursor: "pointer",
    marginRight: "10px",
  },

  cancel: {
    padding: "9px 18px",
    border: "none",
    borderRadius: "6px",
    background: "#6b7280",
    color: "white",
    cursor: "pointer",
  },
};

export default Workers;