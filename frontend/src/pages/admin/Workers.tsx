import { useEffect, useState } from "react";
import { Users } from "lucide-react";
import {
  Btn,
  Card,
  EmptyState,
  PageHeader,
  Pill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

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
      <div style={{ fontFamily: font }}>
        <PageHeader title="Worker Management" />
        <p style={{ color: COLORS.g500 }}>Loading workers…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ fontFamily: font }}>
        <PageHeader title="Worker Management" />
        <EmptyState
          icon={<Users size={40} />}
          title="Couldn't load workers"
          hint={error}
          action={<Btn onClick={loadData}>Retry</Btn>}
        />
      </div>
    );
  }

  return (
    <div style={{ fontFamily: font, maxWidth: 1120 }}>
      <PageHeader
        title="Worker Management"
        subtitle="Manage workers and their ward queues"
      />

      <div style={styles.container}>
        {workers.length === 0 ? (
          <EmptyState
            icon={<Users size={40} />}
            title="No workers found"
          />
        ) : (
          workers.map((worker) => (
            <Card
              key={worker.id}
              style={styles.card}
            >
              <div style={styles.header}>
                <div>
                  <div style={styles.name}>
                    {worker.name}
                  </div>

                  <p style={styles.email}>
                    {worker.email}
                  </p>

                  <Pill
                    color={
                      worker.status === "ACTIVE"
                        ? COLORS.cGreen
                        : COLORS.g500
                    }
                    bg={
                      worker.status === "ACTIVE"
                        ? COLORS.cGreenL
                        : COLORS.g100
                    }
                  >
                    {worker.status}
                  </Pill>
                </div>

                <Btn
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
                >
                  {selectedWorker ===
                  worker.id
                    ? "Close"
                    : "Manage"}
                </Btn>
              </div>

              {/* Assigned Wards */}

              <div style={styles.sectionTitle}>
                Assigned Wards
              </div>

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

                  <div style={styles.sectionTitle}>
                    Assign Ward
                  </div>

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

                  <div style={{ display: "flex", gap: 10 }}>
                    <Btn
                      onClick={
                        handleAssignWard
                      }
                      disabled={
                        assigningWard
                      }
                    >
                      {assigningWard
                        ? "Assigning..."
                        : "Assign Ward"}
                    </Btn>

                    <Btn
                      onClick={
                        closeWorkerPanel
                      }
                      variant="ghost"
                      color={COLORS.g500}
                    >
                      Close
                    </Btn>
                  </div>
                </div>
              )}
            </Card>
          ))
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: 1120,
    display: "grid",
    gridTemplateColumns:
      "repeat(auto-fill, minmax(340px, 1fr))",
    gap: 12,
  },

  card: {
    padding: 20,
  },

  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 10,
  },

  name: {
    fontSize: 16,
    fontWeight: 800,
    color: COLORS.g900,
  },

  email: {
    color: COLORS.g500,
    margin: "5px 0 8px",
    fontSize: 12,
  },

  sectionTitle: {
    fontSize: 13,
    fontWeight: 700,
    color: COLORS.g900,
    margin: "14px 0 8px",
  },

  ward: {
    display: "inline-block",
    padding: "6px 12px",
    marginRight: 8,
    marginBottom: 8,
    background: COLORS.aPurpleL,
    color: COLORS.aPurple,
    borderRadius: 20,
    fontSize: 12,
    fontWeight: 600,
  },

  removeWard: {
    marginLeft: 8,
    padding: 0,
    border: "none",
    background: "transparent",
    color: COLORS.aPurple,
    cursor: "pointer",
    fontSize: 16,
    lineHeight: 1,
  },

  noWard: {
    color: COLORS.g400,
    fontSize: 13,
  },

  count: {
    color: COLORS.g500,
    fontSize: 12,
    marginTop: 8,
  },

  panel: {
    marginTop: 16,
    padding: 16,
    background: COLORS.g50,
    borderRadius: 10,
    border: `1px solid ${COLORS.g200}`,
  },

  select: {
    display: "block",
    width: "100%",
    boxSizing: "border-box" as const,
    padding: 10,
    marginTop: 8,
    marginBottom: 12,
    borderRadius: 8,
    border: `1px solid ${COLORS.g300}`,
    background: "white",
    fontSize: 14,
  },
};

export default Workers;
