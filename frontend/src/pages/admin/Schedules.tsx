import { useEffect, useState } from "react";

import {
  getAdminSchedules,
  createAdminSchedule,
  deleteAdminSchedule,
  getAdminWards,
  getAdminWasteCategories,
} from "../../api/admin";

import type {
  AdminSchedule,
  AdminWard,
  AdminWasteCategory,
} from "../../api/admin";

const days = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
];

function Schedules() {
  const [schedules, setSchedules] = useState<AdminSchedule[]>([]);
  const [wards, setWards] = useState<AdminWard[]>([]);
  const [categories, setCategories] = useState<
    AdminWasteCategory[]
  >([]);

  const [showForm, setShowForm] = useState(false);

  const [wardId, setWardId] = useState("");
  const [categoryId, setCategoryId] = useState("");
  const [day, setDay] = useState("");
  const [startTime, setStartTime] = useState("");
  const [endTime, setEndTime] = useState("");

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      setError("");

      const [
        schedulesResponse,
        wardsResponse,
        categoriesResponse,
      ] = await Promise.all([
        getAdminSchedules(),
        getAdminWards(),
        getAdminWasteCategories(),
      ]);

      if (schedulesResponse.success) {
        setSchedules(schedulesResponse.data);
      }

      if (wardsResponse.success) {
        setWards(wardsResponse.data);
      }

      if (categoriesResponse.success) {
        setCategories(categoriesResponse.data);
      }
    } catch (err: any) {
      console.error(err);

      setError(
        err?.message ||
          "Failed to load schedules."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleAddSchedule(
    e: React.FormEvent
  ) {
    e.preventDefault();

    if (
      !wardId ||
      !categoryId ||
      !day ||
      !startTime ||
      !endTime
    ) {
      alert("Please fill in all fields.");
      return;
    }

    if (startTime >= endTime) {
      alert(
        "End time must be later than start time."
      );
      return;
    }

    const duplicate = schedules.some(
      (schedule) =>
        schedule.ward_id === wardId &&
        schedule.category_id === categoryId &&
        schedule.day === day
    );

    if (duplicate) {
      alert(
        "Duplicate schedule is not allowed."
      );
      return;
    }

    try {
      setSaving(true);
      setError("");

      const response =
        await createAdminSchedule(
          wardId,
          categoryId,
          day,
          startTime,
          endTime
        );

      if (response.success) {
        setSchedules((current) => [
          ...current,
          response.data,
        ]);

        setWardId("");
        setCategoryId("");
        setDay("");
        setStartTime("");
        setEndTime("");

        setShowForm(false);

        alert(
          "Schedule added successfully!"
        );
      }
    } catch (err: any) {
      console.error(err);

      setError(
        err?.message ||
          "Failed to create schedule."
      );
    } finally {
      setSaving(false);
    }
  }

  async function handleDeleteSchedule(
    id: string
  ) {
    const confirmed = window.confirm(
      "Delete this schedule?"
    );

    if (!confirmed) {
      return;
    }

    try {
      setError("");

      await deleteAdminSchedule(id);

      setSchedules((current) =>
        current.filter(
          (schedule) =>
            schedule.id !== id
        )
      );

      alert(
        "Schedule deleted successfully!"
      );
    } catch (err: any) {
      console.error(err);

      setError(
        err?.message ||
          "Failed to delete schedule."
      );
    }
  }

  if (loading) {
    return (
      <div>
        <h1>Collection Schedules</h1>

        <p style={styles.subtitle}>
          Manage waste collection schedules by
          ward.
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
            Manage waste collection schedules
            by ward.
          </p>
        </div>

        <button
          onClick={() =>
            setShowForm(!showForm)
          }
          style={styles.addButton}
        >
          {showForm
            ? "✕ Close"
            : "+ Add Schedule"}
        </button>
      </div>

      {error && (
        <div style={styles.error}>
          {error}

          <button
            onClick={loadData}
            style={styles.retry}
          >
            Retry
          </button>
        </div>
      )}

      {showForm && (
        <form
          onSubmit={handleAddSchedule}
          style={styles.form}
        >
          <h2>
            Add New Schedule
          </h2>

          <div style={styles.formGrid}>
            <div>
              <label style={styles.label}>
                Ward
              </label>

              <select
                value={wardId}
                onChange={(e) =>
                  setWardId(e.target.value)
                }
                style={styles.input}
              >
                <option value="">
                  -- Select Ward --
                </option>

                {wards.map((ward) => (
                  <option
                    key={ward.id}
                    value={ward.id}
                  >
                    {ward.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={styles.label}>
                Waste Category
              </label>

              <select
                value={categoryId}
                onChange={(e) =>
                  setCategoryId(
                    e.target.value
                  )
                }
                style={styles.input}
              >
                <option value="">
                  -- Select Category --
                </option>

                {categories.map(
                  (category) => (
                    <option
                      key={category.id}
                      value={category.id}
                    >
                      {category.name}
                    </option>
                  )
                )}
              </select>
            </div>

            <div>
              <label style={styles.label}>
                Day
              </label>

              <select
                value={day}
                onChange={(e) =>
                  setDay(e.target.value)
                }
                style={styles.input}
              >
                <option value="">
                  -- Select Day --
                </option>

                {days.map((item) => (
                  <option
                    key={item}
                    value={item}
                  >
                    {item}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={styles.label}>
                Start Time
              </label>

              <input
                type="time"
                value={startTime}
                onChange={(e) =>
                  setStartTime(
                    e.target.value
                  )
                }
                style={styles.input}
              />
            </div>

            <div>
              <label style={styles.label}>
                End Time
              </label>

              <input
                type="time"
                value={endTime}
                onChange={(e) =>
                  setEndTime(
                    e.target.value
                  )
                }
                style={styles.input}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={saving}
            style={{
              ...styles.saveButton,
              opacity: saving ? 0.6 : 1,
              cursor: saving
                ? "not-allowed"
                : "pointer",
            }}
          >
            {saving
              ? "Saving..."
              : "Save Schedule"}
          </button>
        </form>
      )}

      <div style={styles.tableWrapper}>
        {schedules.length === 0 ? (
          <div style={styles.empty}>
            <h2>
              No schedules found
            </h2>

            <p>
              Add a collection schedule
              using the button above.
            </p>
          </div>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>
                  Ward
                </th>

                <th style={styles.th}>
                  Category
                </th>

                <th style={styles.th}>
                  Day
                </th>

                <th style={styles.th}>
                  Time
                </th>

                <th style={styles.th}>
                  Action
                </th>
              </tr>
            </thead>

            <tbody>
              {schedules.map(
                (schedule) => (
                  <tr
                    key={schedule.id}
                  >
                    <td style={styles.td}>
                      {schedule.ward}
                    </td>

                    <td style={styles.td}>
                      {schedule.category}
                    </td>

                    <td style={styles.td}>
                      {schedule.day}
                    </td>

                    <td style={styles.td}>
                      {schedule.startTime}
                      {" - "}
                      {schedule.endTime}
                    </td>

                    <td style={styles.td}>
                      <button
                        onClick={() =>
                          handleDeleteSchedule(
                            schedule.id
                          )
                        }
                        style={
                          styles.deleteButton
                        }
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                )
              )}
            </tbody>
          </table>
        )}
      </div>
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

  addButton: {
    padding: "11px 18px",
    border: "none",
    borderRadius: "7px",
    background: "#15803d",
    color: "white",
    cursor: "pointer",
    fontSize: "14px",
  },

  form: {
    background: "white",
    padding: "25px",
    borderRadius: "12px",
    marginBottom: "25px",
    boxShadow:
      "0 4px 15px rgba(0,0,0,0.08)",
    border:
      "1px solid #bbf7d0",
  },

  formGrid: {
    display: "grid",
    gridTemplateColumns:
      "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "20px",
    marginTop: "20px",
  },

  label: {
    display: "block",
    marginBottom: "7px",
    fontWeight: "bold" as const,
  },

  input: {
    width: "100%",
    boxSizing: "border-box" as const,
    padding: "10px",
    border:
      "1px solid #d1d5db",
    borderRadius: "6px",
    background: "white",
  },

  saveButton: {
    marginTop: "25px",
    padding: "11px 20px",
    border: "none",
    borderRadius: "7px",
    background: "#15803d",
    color: "white",
    fontSize: "15px",
    fontWeight: "bold",
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

  deleteButton: {
    padding: "7px 12px",
    border: "none",
    borderRadius: "6px",
    background: "#dc2626",
    color: "white",
    cursor: "pointer",
  },

  loading: {
    padding: "30px",
    background: "white",
    borderRadius: "10px",
    color: "#6b7280",
  },

  empty: {
    padding: "40px",
    textAlign: "center" as const,
    color: "#6b7280",
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

export default Schedules;