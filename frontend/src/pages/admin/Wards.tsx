import { useEffect, useState } from "react";
import { Coins, MapPin } from "lucide-react";
import {
  getAdminDashboard,
  getAdminWards,
  type AdminWard,
} from "../../api/admin";
import {
  Bar,
  Btn,
  Card,
  EmptyState,
  PageHeader,
  Pill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

type WardStat = { ward: string; reports: number; pending: number };

function Wards() {
  const [wards, setWards] = useState<AdminWard[]>([]);
  const [stats, setStats] = useState<WardStat[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function load() {
    try {
      setLoading(true);
      setError("");
      const [wardsRes, dashRes] = await Promise.all([
        getAdminWards(),
        getAdminDashboard().catch(() => null),
      ]);
      if (wardsRes.success) setWards(wardsRes.data);
      if (dashRes?.success) setStats(dashRes.data.wards);
    } catch (err: any) {
      setError(err?.message || "Failed to load wards.");
    } finally {
      setLoading(false);
    }
  }

  const statByName = (name: string) =>
    stats.find((s) => s.ward === name);
  const maxReports = Math.max(
    1,
    ...stats.map((s) => s.reports),
    1
  );

  return (
    <div style={{ fontFamily: font, maxWidth: 1120 }}>
      <PageHeader
        title="Ward Performance"
        subtitle="Activity and backlog per ward"
        right={
          <Btn size="sm" variant="ghost" onClick={load}>
            Refresh
          </Btn>
        }
      />

      {loading && <p style={{ color: COLORS.g500 }}>Loading wards…</p>}

      {!loading && error && (
        <EmptyState
          icon={<MapPin size={40} />}
          title="Couldn't load wards"
          hint={error}
          action={<Btn onClick={load}>Retry</Btn>}
        />
      )}

      {!loading && !error && wards.length === 0 && (
        <EmptyState
          icon={<MapPin size={40} />}
          title="No wards yet"
        />
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
          gap: 12,
        }}
      >
        {wards.map((w) => {
          const st = statByName(w.name);
          const pending = st?.pending || 0;
          const risk =
            pending === 0 ? "low" : pending <= 5 ? "medium" : "high";
          const riskColor =
            risk === "low"
              ? COLORS.cGreen
              : risk === "medium"
                ? COLORS.amber
                : COLORS.red;
          const riskBg =
            risk === "low"
              ? COLORS.cGreenL
              : risk === "medium"
                ? COLORS.amberL
                : COLORS.redL;
          return (
            <Card
              key={w.id}
              style={{ padding: 16, borderLeft: `4px solid ${riskColor}` }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 8,
                  marginBottom: 10,
                }}
              >
                <div>
                  <div
                    style={{ fontSize: 15, fontWeight: 700, color: COLORS.g900 }}
                  >
                    {w.name}
                  </div>
                  <div style={{ fontSize: 11, color: COLORS.g400 }}>
                    {w.zone || "—"}
                    {w.description ? ` · ${w.description}` : ""}
                  </div>
                </div>
                <Pill color={riskColor} bg={riskBg}>
                  {risk === "low"
                    ? "CLEAR"
                    : risk === "medium"
                      ? "WATCH"
                      : "BACKLOG"}
                </Pill>
              </div>
              <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
                {[
                  {
                    v: `${st?.reports ?? 0}`,
                    l: "Reports",
                    c: COLORS.cGreen,
                  },
                  {
                    v: `${pending}`,
                    l: "Pending",
                    c: pending > 0 ? COLORS.red : COLORS.g400,
                  },
                  {
                    v: <Coins size={16} />,
                    l: "Coins flow via completions",
                    c: COLORS.coin,
                  },
                ].map((s) => (
                  <div
                    key={s.l}
                    style={{
                      flex: 1,
                      background: COLORS.g50,
                      borderRadius: 8,
                      padding: "8px 10px",
                      textAlign: "center",
                    }}
                  >
                    <div
                      style={{ fontSize: 18, fontWeight: 800, color: s.c }}
                    >
                      {s.v}
                    </div>
                    <div style={{ fontSize: 10, color: COLORS.g400 }}>
                      {s.l}
                    </div>
                  </div>
                ))}
              </div>
              <Bar
                v={st?.reports ?? 0}
                max={maxReports}
                color={riskColor}
                h={8}
              />
            </Card>
          );
        })}
      </div>
    </div>
  );
}

export default Wards;
