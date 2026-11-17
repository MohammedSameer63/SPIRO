import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  CalendarDays,
  CheckCircle2,
  ClipboardList,
  Hourglass,
  MapPin,
  Truck,
  User,
} from "lucide-react";

import {
  getWorkerStats,
  getWorkerWards,
  getWorkerSchedules,
} from "../../api/reports";
import { getCurrentUser, type User as UserType } from "../../api/auth";

import type {
  WorkerSchedule,
  WorkerWard,
} from "../../api/reports";
import {
  Btn,
  Card,
  EmptyState,
  PageHeader,
  Stat,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

function Dashboard() {
  const navigate = useNavigate();
  const [me, setMe] = useState<UserType | null>(null);
  const [stats, setStats] = useState({
    pending: 0,
    accepted: 0,
    in_progress: 0,
    completed_today: 0,
  });
  const [wards, setWards] = useState<WorkerWard[]>([]);
  const [schedules, setSchedules] = useState<WorkerSchedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadDashboard();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadDashboard() {
    try {
      setLoading(true);
      setError("");
      const [statsResponse, wardsResponse, schedulesResponse, meResponse] =
        await Promise.all([
          getWorkerStats(),
          getWorkerWards(),
          getWorkerSchedules(),
          getCurrentUser().catch(() => null),
        ]);
      if (statsResponse.success) setStats(statsResponse.data);
      if (wardsResponse.success) setWards(wardsResponse.data);
      if (schedulesResponse.success) setSchedules(schedulesResponse.data);
      setMe(meResponse);
    } catch (err: any) {
      setError(err?.message || "Failed to load worker dashboard.");
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div style={{ fontFamily: font }}>
        <PageHeader title="Worker Dashboard" />
        <p style={{ color: COLORS.g500 }}>Loading…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ fontFamily: font }}>
        <PageHeader title="Worker Dashboard" />
        <EmptyState
          icon={<Truck size={40} />}
          title="Couldn't load dashboard"
          hint={error}
          action={<Btn onClick={loadDashboard}>Retry</Btn>}
        />
      </div>
    );
  }

  return (
    <div
      style={{
        fontFamily: font,
        maxWidth: 1120,
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      <PageHeader
        title={`On duty${me ? `, ${me.name.split(" ")[0]}` : ""}`}
        subtitle={
          wards.length > 0
            ? `Covering ${wards.map((w) => w.name).join(" · ")}`
            : "No wards assigned yet"
        }
        right={
          <Btn size="sm" onClick={() => navigate("/worker/queue")}>
            <ClipboardList size={14} /> Open Queue
          </Btn>
        }
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
          gap: 10,
        }}
      >
        <Stat
          label="Pending"
          value={stats.pending}
          icon={<Hourglass size={18} />}
          color={COLORS.amber}
        />
        <Stat
          label="Accepted"
          value={stats.accepted}
          icon={<ClipboardList size={18} />}
          color={COLORS.dBlue}
        />
        <Stat
          label="In Progress"
          value={stats.in_progress}
          icon={<Truck size={18} />}
          color={COLORS.sky}
        />
        <Stat
          label="Done Today"
          value={stats.completed_today}
          icon={<CheckCircle2 size={18} />}
          color={COLORS.cGreen}
        />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: 12,
        }}
      >
        <Card
          style={{
            padding: 20,
            background: "linear-gradient(135deg,#dbeafe,#eff6ff)",
          }}
        >
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: "50%",
              background: COLORS.dBlue,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              marginBottom: 10,
            }}
          >
            <User size={28} />
          </div>
          <div style={{ fontSize: 18, fontWeight: 800, color: COLORS.g900 }}>
            {me?.name || "Worker"}
          </div>
          <div style={{ fontSize: 12, color: COLORS.g500, marginTop: 4 }}>
            {me?.email || ""}
          </div>
          <div style={{ fontSize: 12, color: COLORS.g500 }}>
            {wards.length} ward{wards.length === 1 ? "" : "s"} assigned
          </div>
        </Card>

        <Card style={{ padding: 16 }}>
          <div
            style={{
              fontSize: 14,
              fontWeight: 700,
              color: COLORS.g900,
              marginBottom: 10,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <MapPin size={15} color={COLORS.dBlue} /> My Wards
          </div>
          {wards.length === 0 ? (
            <div style={{ fontSize: 13, color: COLORS.g400 }}>
              No wards assigned — contact your admin.
            </div>
          ) : (
            wards.slice(0, 5).map((w) => (
              <div
                key={w.id}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 13,
                  padding: "6px 0",
                  borderBottom: `1px solid ${COLORS.g100}`,
                }}
              >
                <span style={{ fontWeight: 600, color: COLORS.g900 }}>
                  {w.name}
                </span>
                <span style={{ color: COLORS.g400 }}>{w.zone || ""}</span>
              </div>
            ))
          )}
        </Card>

        <Card style={{ padding: 16 }}>
          <div
            style={{
              fontSize: 14,
              fontWeight: 700,
              color: COLORS.g900,
              marginBottom: 10,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <CalendarDays size={15} color={COLORS.dBlue} /> Upcoming
            Collections
          </div>
          {schedules.length === 0 ? (
            <div style={{ fontSize: 13, color: COLORS.g400 }}>
              No collection schedules for your wards.
            </div>
          ) : (
            schedules.slice(0, 5).map((s) => (
              <div
                key={s.id}
                style={{ fontSize: 13, padding: "6px 0", color: COLORS.g700 }}
              >
                <strong>{s.day}</strong> · {s.ward} · {s.category}
                <div style={{ fontSize: 11, color: COLORS.g400 }}>
                  {s.startTime} – {s.endTime}
                </div>
              </div>
            ))
          )}
        </Card>
      </div>
    </div>
  );
}

export default Dashboard;
