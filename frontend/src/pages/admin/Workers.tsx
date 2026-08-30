import { useEffect, useState } from "react";

import {
  getAdminWorkers,
  getAdminWards,
  assignWorkerToWard,
  getAdminPendingReports,
  assignReportToWorker,
} from "../../api/admin";

import type {
  AdminWorker,
  AdminWard,
} from "../../api/admin";

type PendingReport = {
  id: string;
  description: string | null;
  address: string | null;
  latitude: number;
  longitude: number;
  status:
    | "PENDING"
    | "ACCEPTED"
    | "IN_PROGRESS"
    | "COMPLETED";
  image_url: string;
  created_at: string;
  updated_at: string;
};

function Workers() {
  const [workers, setWorkers] =
    useState<AdminWorker[]>([]);

  const [wards, setWards] =
    useState<AdminWard[]>([]);

  const [reports, setReports] =
    useState<PendingReport[]>([]);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  const [selectedWorker, setSelectedWorker] =
    useState<string | null>(null);

  const [selectedWard, setSelectedWard] =
    useState("");

  const [selectedReport, setSelectedReport] =
    useState("");

  const [assigningWard, setAssigningWard] =
    useState(false);

  const [assigningReport, setAssigningReport] =
    useState(false);

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
        reportsResponse,
      ] = await Promise.all([
        getAdminWorkers(),
        getAdminWards(),
        getAdminPendingReports(),
      ]);

      if (workersResponse.success) {
        setWorkers(workersResponse.data);
      }

      if (wardsResponse.success) {
        setWards(wardsResponse.data);
      }

      if (reportsResponse.success) {
        setReports(reportsResponse.data);
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

  async function handleAssignReport() {
    if (!selectedWorker) {
      return;
    }

    if (!selectedReport) {
      alert("Please select a report.");
      return;
    }

    try {
      setAssigningReport(true);

      await assignReportToWorker(
        selectedReport,
        selectedWorker
      );

      alert(
        "Report assigned to worker successfully!"
      );

      setSelectedReport("");

      await loadData();
    } catch (err: any) {
      console.error(err);

      alert(
        err?.message ||
          "Failed to assign report."
      );
    } finally {
      setAssigningReport(false);
    }
  }

  function closeWorkerPanel() {
    setSelectedWorker(null);
    setSelectedWard("");
    setSelectedReport("");
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
        Manage workers, wards and report assignments.
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
                      setSelectedReport("");
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
                      </span>
                    )
                  )}
                </div>
              )}

              <p style={styles.count}>
                {worker.wards.length}/5
                wards assigned
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

                  {/* Assign Report */}

                  <h3
                    style={
                      styles.reportTitle
                    }
                  >
                    Assign Pending Report
                  </h3>

                  {reports.length ===
                  0 ? (
                    <p
                      style={
                        styles.noReports
                      }
                    >
                      No unassigned pending
                      reports available.
                    </p>
                  ) : (
                    <>
                      <select
                        value={
                          selectedReport
                        }
                        onChange={(e) =>
                          setSelectedReport(
                            e.target.value
                          )
                        }
                        style={
                          styles.select
                        }
                      >
                        <option value="">
                          -- Select Pending Report --
                        </option>

                        {reports.map(
                          (report) => (
                            <option
                              key={report.id}
                              value={
                                report.id
                              }
                            >
                              {report.description ||
                                "Waste Report"}{" "}
                              —{" "}
                              {report.address ||
                                "No address"}
                            </option>
                          )
                        )}
                      </select>

                      <button
                        onClick={
                          handleAssignReport
                        }
                        disabled={
                          assigningReport
                        }
                        style={
                          styles.reportButton
                        }
                      >
                        {assigningReport
                          ? "Assigning..."
                          : "Assign Report"}
                      </button>
                    </>
                  )}

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