import { useEffect, useState } from "react";
import { CalendarDays, Plus, Trash2 } from "lucide-react";

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
import {
  Btn,
  Card,
  EmptyState,
  PageHeader,
  WastePill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

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
  const [categories, setCategories] = useState<AdminWasteCategory[]>([]);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      setError("");
      const [schedulesResponse, wardsResponse, categoriesResponse] =
        await Promise.all([
          getAdminSchedules(),
          getAdminWards(),
          getAdminWasteCategories(),
        ]);
      if (schedulesResponse.success) setSchedules(schedulesResponse.data);
      if (wardsResponse.success) setWards(wardsResponse.data);
      if (categoriesResponse.success) setCategories(categoriesResponse.data);
    } catch (err: any) {
      setError(err?.message || "Failed to load schedules.");
    } finally {
      setLoading(false);
    }
  }

  async function handleAddSchedule(e: React.FormEvent) {
    e.preventDefault();
    if (!wardId || !categoryId || !day || !startTime || !endTime) {
      alert("Please fill in all fields.");
      return;
    }
    if (startTime >= endTime) {
      alert("End time must be later than start time.");
      return;
    }
    const duplicate = schedules.some(
      (s) =>
        s.ward_id === wardId && s.category_id === categoryId && s.day === day
    );
    if (duplicate) {
      alert("Duplicate schedule is not allowed.");
      return;
    }
    try {
      setSaving(true);
      setError("");
      const response = await createAdminSchedule(
        wardId,
        categoryId,
        day,
        startTime,
        endTime
      );
      if (response.success) {
        setSchedules((cur) => [...cur, response.data]);
        setWardId("");
        setCategoryId("");
        setDay("");
        setStartTime("");
        setEndTime("");
        setShowForm(false);
      }
    } catch (err: any) {
      setError(err?.message || "Failed to create schedule.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDeleteSchedule(id: string) {
    if (!window.confirm("Delete this schedule?")) return;
    try {
      setError("");
      await deleteAdminSchedule(id);
      setSchedules((cur) => cur.filter((s) => s.id !== id));
    } catch (err: any) {
      setError(err?.message || "Failed to delete schedule.");
    }
  }

  return (
    <div style={{ fontFamily: font, maxWidth: 1120 }}>
      <PageHeader
        title="Collection Schedules"
        subtitle="Pickup timetable by ward and stream"
        right={
          <Btn size="sm" onClick={() => setShowForm(!showForm)}>
            <Plus size={14} /> {showForm ? "Close" : "Add Schedule"}
          </Btn>
        }
      />

      {loading && <p style={{ color: COLORS.g500 }}>Loading schedules…</p>}

      {!loading && error && (
        <Card
          style={{
            padding: 12,
            marginBottom: 12,
            background: COLORS.redL,
            fontSize: 13,
            color: COLORS.red,
          }}
        >
          {error}
        </Card>
      )}

      {showForm && (
        <Card style={{ padding: 20, marginBottom: 14 }}>
          <div
            style={{
              fontSize: 15,
              fontWeight: 800,
              color: COLORS.g900,
              marginBottom: 14,
            }}
          >
            Add New Schedule
          </div>
          <form onSubmit={handleAddSchedule}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: 12,
              }}
            >
              {(
                [
                  ["Ward", wardId, setWardId, wards.map((w) => ({ id: w.id, name: w.name })), "-- Select Ward --"],
                  ["Waste Category", categoryId, setCategoryId, categories.map((c) => ({ id: c.id, name: c.name })), "-- Select Category --"],
                  ["Day", day, setDay, days.map((d) => ({ id: d, name: d })), "-- Select Day --"],
                ] as const
              ).map(([label, value, set, opts, placeholder]) => (
                <div key={label}>
                  <label style={labelStyle}>{label}</label>
                  <select
                    value={value}
                    onChange={(e) => set(e.target.value)}
                    style={inputStyle}
                  >
                    <option value="">{placeholder}</option>
                    {opts.map((o) => (
                      <option key={o.id} value={o.id}>
                        {o.name}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
              <div>
                <label style={labelStyle}>Start Time</label>
                <input
                  type="time"
                  value={startTime}
                  onChange={(e) => setStartTime(e.target.value)}
                  style={inputStyle}
                />
              </div>
              <div>
                <label style={labelStyle}>End Time</label>
                <input
                  type="time"
                  value={endTime}
                  onChange={(e) => setEndTime(e.target.value)}
                  style={inputStyle}
                />
              </div>
            </div>
            <Btn type="submit" disabled={saving} style={{ marginTop: 16 }}>
              {saving ? "Saving…" : "Save Schedule"}
            </Btn>
          </form>
        </Card>
      )}

      {!loading && !error && schedules.length === 0 && (
        <EmptyState
          icon={<CalendarDays size={40} />}
          title="No schedules found"
          hint="Add a collection schedule using the button above."
        />
      )}

      {schedules.length > 0 && (
        <Card style={{ padding: 0, overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 640 }}>
            <thead>
              <tr>
                {["Ward", "Category", "Day", "Time", ""].map((h) => (
                  <th
                    key={h}
                    style={{
                      padding: "12px 15px",
                      textAlign: "left",
                      background: COLORS.g50,
                      borderBottom: `1px solid ${COLORS.g200}`,
                      fontSize: 11,
                      textTransform: "uppercase",
                      letterSpacing: 0.6,
                      color: COLORS.g500,
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {schedules.map((s) => (
                <tr key={s.id}>
                  <td style={tdStyle}>
                    <strong>{s.ward}</strong>
                  </td>
                  <td style={tdStyle}>
                    <WastePill category={s.category} />
                  </td>
                  <td style={tdStyle}>{s.day}</td>
                  <td style={tdStyle}>
                    {s.startTime} – {s.endTime}
                  </td>
                  <td style={{ ...tdStyle, textAlign: "right" }}>
                    <Btn
                      size="sm"
                      color={COLORS.red}
                      onClick={() => handleDeleteSchedule(s.id)}
                    >
                      <Trash2 size={13} /> Delete
                    </Btn>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: "block",
  marginBottom: 6,
  fontWeight: 600,
  fontSize: 12,
  color: COLORS.g600,
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  boxSizing: "border-box",
  padding: 10,
  border: `1px solid ${COLORS.g300}`,
  borderRadius: 8,
  background: "white",
  fontSize: 14,
};

const tdStyle: React.CSSProperties = {
  padding: "12px 15px",
  borderBottom: `1px solid ${COLORS.g200}`,
  fontSize: 13,
};

export default Schedules;
