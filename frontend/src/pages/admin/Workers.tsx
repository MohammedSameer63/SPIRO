import { useEffect, useState } from "react";

import {
  getAdminWorkers,
  getAdminWards,
  assignWorkerToWard,
  removeWorkerFromWard,
} from "../../api/admin";

import type {
  AdminWorker,
  AdminWard,
} from "../../api/admin";

function Workers() {
  const [workers, setWorkers] =
    useState<AdminWorker[]>([]);

  const [wards, setWards] =
    useState<AdminWard[]>([]);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  const [selectedWorker, setSelectedWorker] =
    useState<string | null>(null);

  const [selectedWard, setSelectedWard] =
    useState("");

  const [assigningWard, setAssigningWard] =
    useState(false);

  const [removingWardId, setRemovingWardId] =
    useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      setError("");

      const [
        workersResponse,
        wardsResponse,
      ] = await Promise.all([
        getAdminWorkers(),
        getAdminWards(),
      ]);

      if (workersResponse.success) {
        setWorkers(workersResponse.data);
      }

      if (wardsResponse.success) {
        setWards(wardsResponse.data);
      }

    } catch (err: any) {
      console.error(err);

      setError(
        err?.message ||
          "Failed to load worker management data."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleAssignWard() {
    if (!selectedWorker) {
      return;
    }

    if (!selectedWard) {
      alert("Please select a ward.");
      return;
    }

    const worker =
      workers.find(
        (item) =>
          item.id === selectedWorker
      );

    if (!worker) {
      return;
    }

    if (worker.wards.length >= 5) {
      alert(
        "A worker can be assigned to a maximum of 5 wards."
      );
      return;
    }

    if (
      worker.wards.some(
        (ward) =>
          ward.id === selectedWard
      )
    ) {
      alert(
        "Worker is already assigned to this ward."
      );
      return;
    }

    try {
      setAssigningWard(true);

      await assignWorkerToWard(
        selectedWorker,
        selectedWard
      );

      alert(
        "Worker assigned to ward successfully!"
      );

      setSelectedWard("");

      await loadData();
    } catch (err: any) {
      console.error(err);

      alert(
        err?.message ||
          "Failed to assign worker to ward."
      );
    } finally {
      setAssigningWard(false);
    }
  }

  function closeWorkerPanel() {
    setSelectedWorker(null);
    setSelectedWard("");
  }

  async function handleRemoveWard(
    workerId: string,
    wardId: string,
    wardName: string
  ) {
    if (!window.confirm(`Remove ${wardName} from this worker?`)) {
      return;
    }

    try {
      setRemovingWardId(wardId);
      await removeWorkerFromWard(workerId, wardId);

      setWorkers((current) =>
        current.map((worker) =>
          worker.id === workerId
            ? {
                ...worker,
                wards: worker.wards.filter(
                  (ward) => ward.id !== wardId
                ),
              }
            : worker
        )
      );
    } catch (err: any) {
      console.error(err);
      alert(
        err?.message ||
          "Failed to remove worker from ward."
      );
    } finally {
      setRemovingWardId(null);
    }
  }

  if (loading) {
    return (
      <div>
        <h1>Worker Management</h1>
        <p>Loading workers...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <h1>Worker Management</h1>

        <div style={styles.error}>
          {error}

          <button
            onClick={loadData}
            style={styles.retry}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1>
        Worker Management
      </h1>

      <p style={styles.subtitle}>
        Manage workers and their ward queues.
      </p>

      <div style={styles.container}>
        {workers.length === 0 ? (
          <div style={styles.empty}>
            No workers found.
          </div>
        ) : (
          workers.map((worker) => (
            <div
              key={worker.id}
              style={styles.card}
            >
              <div style={styles.header}>
                <div>
                  <h2>
                    {worker.name}
                  </h2>

                  <p style={styles.email}>
                    {worker.email}
                  </p>

                  <p style={styles.status}>
                    Status: {worker.status}
                  </p>
                </div>

                <button
                  onClick={() => {
                    if (
                      selectedWorker ===
                      worker.id
                    ) {
                      closeWorkerPanel();
                    } else {
                      setSelectedWorker(
                        worker.id
                      );
                      setSelectedWard("");
                    }
                  }}
                  style={styles.button}
                >
                  {selectedWorker ===
                  worker.id
                    ? "Close"
                    : "Manage"}
                </button>
              </div>

              {/* Assigned Wards */}

              <h3>
                Assigned Wards
              </h3>

              {worker.wards.length ===
              0 ? (
                <p style={styles.noWard}>
                  No wards assigned.
                </p>
              ) : (
                <div>
                  {worker.wards.map(
                    (ward) => (
                      <span
                        key={ward.id}
                        style={styles.ward}
                      >
                        {ward.name}
                        {selectedWorker === worker.id && (
                          <button
                            aria-label={`Remove ${ward.name}`}
                            disabled={
                              removingWardId === ward.id
                            }
                            onClick={() =>
                              handleRemoveWard(
                                worker.id,
                                ward.id,
                                ward.name
                              )
                            }
                            style={styles.removeWard}
                          >
                            {removingWardId === ward.id
                              ? "…"
                              : "×"}
                          </button>
                        )}
                      </span>
                    )
                  )}
                </div>
              )}

              <p style={styles.count}>
                {new Set(
                  worker.wards.map((ward) => ward.id)
                ).size} {" "}
                {worker.wards.length === 1
                  ? "ward"
                  : "wards"} assigned
              </p>

              {/* Management Panel */}

              {selectedWorker ===
                worker.id && (
                <div style={styles.panel}>
                  {/* Assign Ward */}

                  <h3>
                    Assign Ward
                  </h3>

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

                    {wards.map(
                      (ward) => {
                        const alreadyAssigned =
                          worker.wards.some(
                            (item) =>
                              item.id ===
                              ward.id
                          );

                        return (
                          <option
                            key={ward.id}
                            value={ward.id}
                            disabled={
                              alreadyAssigned
                            }
                          >
                            {ward.name}
                            {alreadyAssigned
                              ? " (Already assigned)"
                              : ""}
                          </option>
                        );
                      }
                    )}
                  </select>

                  <button
                    onClick={
                      handleAssignWard
                    }
                    disabled={
                      assigningWard
                    }
                    style={
                      styles.confirm
                    }
                  >
                    {assigningWard
                      ? "Assigning..."
                      : "Assign Ward"}
                  </button>

                  <button
                    onClick={
                      closeWorkerPanel
                    }
                    style={
                      styles.cancel
                    }
                  >
                    Close
                  </button>
                </div>
              )}
            </div>
          ))
        )}
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
    maxWidth: "950px",
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
    justifyContent:
      "space-between",
    alignItems: "center",
  },

  email: {
    color: "#6b7280",
    margin: "5px 0",
  },

  status: {
    color: "#6b7280",
    fontSize: "13px",
  },

  button: {
    padding: "10px 16px",
    border: "none",
    borderRadius: "7px",
    background: "#15803d",
    color: "white",
    cursor: "pointer",
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

  removeWard: {
    marginLeft: "8px",
    padding: "0",
    border: "none",
    background: "transparent",
    color: "#166534",
    cursor: "pointer",
    fontSize: "16px",
    lineHeight: "1",
  },

  noWard: {
    color: "#9ca3af",
  },

  count: {
    color: "#6b7280",
    fontSize: "13px",
  },

  panel: {
    marginTop: "20px",
    padding: "20px",
    background: "#f0fdf4",
    borderRadius: "10px",
    border:
      "1px solid #bbf7d0",
  },

  select: {
    display: "block",
    width: "100%",
    maxWidth: "600px",
    padding: "10px",
    marginTop: "8px",
    marginBottom: "15px",
    borderRadius: "6px",
    border:
      "1px solid #d1d5db",
    background: "white",
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

  reportTitle: {
    marginTop: "30px",
  },

  reportButton: {
    padding: "9px 18px",
    border: "none",
    borderRadius: "6px",
    background: "#2563eb",
    color: "white",
    cursor: "pointer",
    marginRight: "10px",
  },

  cancel: {
    marginTop: "25px",
    padding: "9px 18px",
    border: "none",
    borderRadius: "6px",
    background: "#6b7280",
    color: "white",
    cursor: "pointer",
  },

  noReports: {
    color: "#6b7280",
  },

  empty: {
    padding: "30px",
    background: "white",
    borderRadius: "10px",
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

export default Workers;
