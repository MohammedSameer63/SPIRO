import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Brain,
  ClipboardList,
  Coins,
  LayoutDashboard,
  Users,
} from "lucide-react";
import {
  getAdminDashboard,
  getAdminReports,
  type AdminDashboardResponse,
} from "../../api/admin";
import {
  Bar,
  Btn,
  Card,
  EmptyState,
  PageHeader,
  Stat,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

type DashboardData = AdminDashboardResponse["data"];

function Dashboard() {
  const navigate = useNavigate();
  const [data, setData] = useState<DashboardData | null>(null);
  const [coinsIssued, setCoinsIssued] = useState(0);
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
      const [response, reports] = await Promise.all([
        getAdminDashboard(),
        getAdminReports(undefined, 500).catch(() => null),
      ]);
      if (response.success) setData(response.data);
      if (reports?.success) {
        setCoinsIssued(
          reports.data.reduce(
            (sum, r) => sum + (Number(r.coins_awarded) || 0),
            0
          )
        );
      }
    } catch (err: any) {
      setError(err?.message || "Failed to load dashboard.");
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div style={{ fontFamily: font }}>
        <PageHeader title="Today's Operations" />
        <p style={{ color: COLORS.g500 }}>Loading…</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div style={{ fontFamily: font }}>
        <PageHeader title="Today's Operations" />
        <EmptyState
          icon={<LayoutDashboard size={40} />}
          title="Couldn't load dashboard"
          hint={error}
          action={<Btn onClick={loadDashboard}>Retry</Btn>}
        />
      </div>
    );
  }

  const maxWardReports = Math.max(
    1,
    ...data.wards.map((w) => w.reports)
  );

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
        title="Today's Operations"
        subtitle="Municipal overview across wards, workers and AI"
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
          gap: 10,
        }}
      >
        <Stat
          label="Citizens"
          value={data.stats.total_citizens}
          icon={<Users size={18} />}
          color={COLORS.cGreen}
        />
        <Stat
          label="Workers"
          value={data.stats.total_workers}
          icon={<Users size={18} />}
          color={COLORS.dBlue}
        />
        <Stat
          label="Reports"
          value={data.stats.total_reports}
          icon={<ClipboardList size={18} />}
          color={COLORS.aPurple}
        />
        <Stat
          label="Pending"
          value={data.stats.pending_reports}
          icon={<ClipboardList size={18} />}
          color={COLORS.amber}
        />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: 12,
        }}
      >
        <Card style={{ padding: 16 }}>
          <div
            style={{
              fontSize: 14,
              fontWeight: 700,
              color: COLORS.g900,
              marginBottom: 12,
            }}
          >
            Ward Activity
          </div>
          {data.wards.length === 0 && (
            <div style={{ fontSize: 13, color: COLORS.g400 }}>
              No wards yet.
            </div>
          )}
          {data.wards.map((w) => (
            <div key={w.ward} style={{ marginBottom: 12 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: 4,
                }}
              >
                <span
                  style={{
                    fontSize: 12,
                    color: COLORS.g700,
                    fontWeight: 600,
                  }}
                >
                  {w.ward}
                </span>
                <span style={{ fontSize: 12, color: COLORS.g500 }}>
                  {w.reports} reports · {w.pending} pending
                </span>
              </div>
              <Bar v={w.reports} max={maxWardReports} color={COLORS.aPurple} h={8} />
            </div>
          ))}
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Card style={{ padding: 16 }}>
            <div
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: COLORS.g900,
                marginBottom: 4,
              }}
            >
              Green Coins Issued
            </div>
            <div
              style={{ fontSize: 36, fontWeight: 900, color: COLORS.cGreen }}
            >
              <Coins size={30} /> {coinsIssued.toLocaleString()}
            </div>
            <div style={{ fontSize: 12, color: COLORS.g500, marginTop: 4 }}>
              Across all completed collections
            </div>
          </Card>

          <Card style={{ padding: 16 }}>
            <div
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: COLORS.g900,
                marginBottom: 10,
              }}
            >
              Worker Load
            </div>
            {data.workers.length === 0 && (
              <div style={{ fontSize: 13, color: COLORS.g400 }}>
                No workers registered.
              </div>
            )}
            {data.workers.slice(0, 6).map((w) => (
              <div
                key={w.id}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 12,
                  padding: "6px 0",
                  borderBottom: `1px solid ${COLORS.g100}`,
                }}
              >
                <span style={{ fontWeight: 600, color: COLORS.g900 }}>
                  {w.name}
                </span>
                <span style={{ color: COLORS.g500 }}>
                  {w.active_reports} active · {w.completed_today} done today
                </span>
              </div>
            ))}
            <Btn
              variant="ghost"
              size="sm"
              onClick={() => navigate("/admin/workers")}
              style={{ marginTop: 10 }}
            >
              Manage workers →
            </Btn>
          </Card>
        </div>
      </div>

      <Card
        style={{ padding: 16, cursor: "pointer" }}
        onClick={() => navigate("/admin/ml")}
      >
        <div
          style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, fontWeight: 700, color: COLORS.g900 }}
        >
          <Brain size={16} color={COLORS.aPurple} /> AI Pipeline: live — see
          model registry →
        </div>
      </Card>

      <Card style={{ padding: 16 }}>
        <div
          style={{
            fontSize: 14,
            fontWeight: 700,
            color: COLORS.g900,
            marginBottom: 4,
          }}
        >
          Live Feed
        </div>
        <div style={{ fontSize: 12, color: COLORS.g500, marginBottom: 8 }}>
          Open the audit trail for the full per-report timeline.
        </div>
        <Btn variant="ghost" size="sm" onClick={() => navigate("/admin/audit")}>
          Open Audit Trail →
        </Btn>
      </Card>

    </div>
  );
}

export default Dashboard;
