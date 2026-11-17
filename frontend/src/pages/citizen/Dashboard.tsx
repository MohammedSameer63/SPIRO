import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Bell,
  Camera,
  ClipboardList,
  Coins,
  Hourglass,
  Leaf,
} from "lucide-react";
import { getMyReports, type Report } from "../../api/reports";
import { getWallet, type CoinWallet } from "../../api/coins";
import { getCurrentUser, type User } from "../../api/auth";
import {
  Btn,
  Card,
  CoinTag,
  EmptyState,
  PageHeader,
  Stat,
  StatusPill,
  WastePill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

function Dashboard() {
  const navigate = useNavigate();
  const [user, setUser] = useState<User | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [wallet, setWallet] = useState<CoinWallet | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadAll() {
    try {
      setLoading(true);
      setError("");
      const [me, myReports] = await Promise.all([
        getCurrentUser().catch(() => null),
        getMyReports(),
      ]);
      setUser(me);
      setReports(myReports);
      try {
        setWallet(await getWallet());
      } catch {
        setWallet(null);
      }
    } catch (err: any) {
      setError(err?.message || "Failed to load dashboard.");
    } finally {
      setLoading(false);
    }
  }

  const pending = reports.filter((r) => r.status === "PENDING").length;
  const completed = reports.filter((r) => r.status === "COMPLETED").length;
  const recent = reports.slice(0, 4);

  if (loading) {
    return (
      <div style={{ fontFamily: font }}>
        <PageHeader title="Dashboard" />
        <p style={{ color: COLORS.g500 }}>Loading your overview…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ fontFamily: font }}>
        <PageHeader title="Dashboard" />
        <EmptyState
          icon={<Bell size={40} />}
          title="Couldn't load dashboard"
          hint={error}
          action={<Btn onClick={loadAll}>Retry</Btn>}
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
      <Card
        style={{
          padding: 20,
          borderLeft: `4px solid ${COLORS.cGreen}`,
          background: "linear-gradient(135deg,#f0fdf4,#dcfce7)",
        }}
      >
        <div
          style={{ display: "flex", alignItems: "center", gap: 14 }}
        >
          <span style={{ fontSize: 40 }}>
            <Leaf size={40} color={COLORS.cGreenD} />
          </span>
          <div>
            <div
              style={{
                fontSize: 10,
                fontWeight: 700,
                color: COLORS.cGreen,
                letterSpacing: 1,
                textTransform: "uppercase",
                marginBottom: 2,
              }}
            >
              Good {daypart()}, {user?.name?.split(" ")[0] || "Citizen"}
            </div>
            <div
              style={{
                fontSize: 18,
                fontWeight: 800,
                color: COLORS.cGreenD,
              }}
            >
              Keep your ward clean
            </div>
            <div
              style={{ fontSize: 12, color: COLORS.g600, marginTop: 2 }}
            >
              Snap waste → AI classifies it → worker collects → you earn
              Green Coins
            </div>
          </div>
        </div>
        <Btn
          onClick={() => navigate("/citizen/report")}
          style={{ width: "100%", justifyContent: "center", marginTop: 12 }}
        >
          <Camera size={16} /> Submit Waste Now
        </Btn>
      </Card>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: 10,
        }}
      >
        <Stat
          label="Green Coins"
          value={wallet ? wallet.balance : "—"}
          icon={<Coins size={18} />}
          color={COLORS.coin}
        />
        <Stat
          label="Completed"
          value={completed}
          icon={<ClipboardList size={18} />}
          color={COLORS.dBlue}
        />
        <Stat
          label="Pending"
          value={pending}
          icon={<Hourglass size={18} />}
          color={COLORS.amber}
        />
      </div>

      <Card style={{ padding: 16 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 12,
          }}
        >
          <span
            style={{ fontSize: 15, fontWeight: 700, color: COLORS.g900 }}
          >
            Recent Reports
          </span>
          <span
            onClick={() => navigate("/citizen/reports")}
            style={{
              fontSize: 12,
              color: COLORS.cGreen,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            See all →
          </span>
        </div>
        {recent.length === 0 ? (
          <div style={{ fontSize: 13, color: COLORS.g400 }}>
            No reports yet — submit your first one above.
          </div>
        ) : (
          recent.map((r, i) => (
            <div
              key={r.id}
              onClick={() => navigate(`/citizen/reports/${r.id}`)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "8px 0",
                borderBottom:
                  i < recent.length - 1
                    ? `1px solid ${COLORS.g100}`
                    : "none",
                cursor: "pointer",
              }}
            >
              <div style={{ flex: 1 }}>
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 600,
                    color: COLORS.g900,
                  }}
                >
                  {r.description || "Waste report"}
                </div>
                <div style={{ fontSize: 11, color: COLORS.g400 }}>
                  {new Date(r.created_at).toLocaleDateString()} ·{" "}
                  {r.prediction?.category_name || "Analyzing…"}
                </div>
              </div>
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "flex-end",
                  gap: 3,
                }}
              >
                <StatusPill status={r.status} />
                {r.prediction && (
                  <WastePill category={r.prediction.category_name} />
                )}
              </div>
            </div>
          ))
        )}
      </Card>

      {wallet && wallet.transactions.length > 0 && (
        <Card style={{ padding: 16 }}>
          <div
            style={{
              fontSize: 14,
              fontWeight: 700,
              color: COLORS.g900,
              marginBottom: 8,
            }}
          >
            Latest Coin Activity
          </div>
          {wallet.transactions.slice(0, 3).map((tx) => (
            <div
              key={tx.id}
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 12,
                color: COLORS.g600,
                padding: "6px 0",
              }}
            >
              <span>{tx.note || tx.reason}</span>
              <CoinTag amount={Number(tx.amount) || 0} />
            </div>
          ))}
        </Card>
      )}
    </div>
  );
}

function daypart(): string {
  const h = new Date().getHours();
  if (h < 12) return "morning";
  if (h < 17) return "afternoon";
  return "evening";
}

export default Dashboard;
