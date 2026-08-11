import { useState } from "react";

type Schedule = {
  id: string;
  ward: string;
  category: string;
  day: string;
  startTime: string;
  endTime: string;
};

const initialSchedules: Schedule[] = [
  {
    id: "schedule-001",
    ward: "Ward 12",
    category: "Plastic",
    day: "Monday",
    startTime: "08:00",
    endTime: "12:00",
  },
  {
    id: "schedule-002",
    ward: "Ward 12",
    category: "Organic",
    day: "Wednesday",
    startTime: "07:00",
    endTime: "11:00",
  },
  {
    id: "schedule-003",
    ward: "Ward 15",
    category: "Paper",
    day: "Friday",
    startTime: "09:00",
    endTime: "13:00",
  },
];

const wards = [
  "Ward 12",
  "Ward 15",
  "Ward 18",
  "Ward 20",
];

const categories = [
  "Plastic",
  "Paper",
  "Organic",
];

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
  const [schedules, setSchedules] =
    useState(initialSchedules);

  const [showForm, setShowForm] =
    useState(false);

  const [ward, setWard] =
    useState("");

  const [category, setCategory] =
    useState("");

  const [day, setDay] =
    useState("");

  const [startTime, setStartTime] =
    useState("");

  const [endTime, setEndTime] =
    useState("");

  const handleAddSchedule = (
    e: React.FormEvent
  ) => {
    e.preventDefault();

    if (
      !ward ||
      !category ||
      !day ||
      !startTime ||
      !endTime
    ) {
      alert(
        "Please fill in all fields."
      );

      return;
    }

    const duplicate =
      schedules.some(
        (schedule) =>
          schedule.ward === ward &&
          schedule.category === category &&
          schedule.day === day
      );

    if (duplicate) {
      alert(
        "Duplicate schedule is not allowed."
      );

      return;
    }

    const newSchedule: Schedule = {
      id:
        "schedule-" +
        Date.now(),

      ward,
      category,
      day,
      startTime,
      endTime,
    };

    setSchedules([
      ...schedules,
      newSchedule,
    ]);

    // Reset form

    setWard("");
    setCategory("");
    setDay("");
    setStartTime("");
    setEndTime("");

    setShowForm(false);

    alert(
      "Schedule added successfully!"
    );
  };

  const deleteSchedule = (
    id: string
  ) => {
    const confirmed =
      window.confirm(
        "Delete this schedule?"
      );

    if (!confirmed) {
      return;
    }

    setSchedules(
      schedules.filter(
        (schedule) =>
          schedule.id !== id
      )
    );
  };

  return (
    <div>
      <div style={styles.header}>
        <div>
          <h1>
            Collection Schedules
          </h1>

          <p style={styles.subtitle}>
            Manage waste collection schedules by ward.
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

      {/* Add Schedule Form */}

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
                value={ward}
                onChange={(e) =>
                  setWard(e.target.value)
                }
                style={styles.input}
              >
                <option value="">
                  -- Select Ward --
                </option>

                {wards.map((item) => (
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
                Waste Category
              </label>

              <select
                value={category}
                onChange={(e) =>
                  setCategory(
                    e.target.value
                  )
                }
                style={styles.input}
              >
                <option value="">
                  -- Select Category --
                </option>

                {categories.map(
                  (item) => (
                    <option
                      key={item}
                      value={item}
                    >
                      {item}
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
            style={styles.saveButton}
          >
            Save Schedule
          </button>
        </form>
      )}

      {/* Schedule Table */}

      <div style={styles.tableWrapper}>

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
                <tr key={schedule.id}>

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
                        deleteSchedule(
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
    cursor: "pointer",
    fontSize: "15px",
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

  deleteButton: {
    padding: "7px 12px",
    border: "none",
    borderRadius: "6px",
    background: "#dc2626",
    color: "white",
    cursor: "pointer",
  },
};

export default Schedules;