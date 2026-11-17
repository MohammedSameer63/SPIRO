import { useEffect, useState } from "react";
import {
  BadgeCheck,
  Bell,
  Check,
  Coins,
  Sparkles,
  Truck,
} from "lucide-react";
import { getMyReports, type Report } from "../../api/reports";
import { getWallet, type CoinTransaction } from "../../api/coins";
import { Btn, EmptyState, PageHeader } from "../../components/ui";
import { COLORS, font } from "../../theme";

type Alert = {
  id: string;
  kind: "reward" | "verified" | "ai" | "schedule";
  title: string;
  msg: string;
  time: string;
  ts: number;
};

const KIND_META = {
  reward: { color: COLORS.cGreen, icon: <Coins size={18} /> },
  verified: { color: COLORS.dBlue, icon: <BadgeCheck size={18} /> },
  ai: { color: COLORS.aPurple, icon: <Sparkles size={18} /> },
  schedule: { color: COLORS.sky, icon: <Truck size={18} /> },
} as const;

const READ_KEY = "spiro_alerts_read";

function readIds(): string[] {
  try {
    return JSON.parse(localStorage.getItem(READ_KEY) || "[]");
  } catch {
    return [];
  }
}

function Alerts() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [read, setRead] = useState<string[]>(readIds());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    load();
  }, []);

  async function load() {
    try {
      setLoading(true);
      setError("");
      const [reports, wallet] = await Promise.all([
        getMyReports(),
        getWallet().catch(() => null),
      ]);
      setAlerts(buildAlerts(reports, wallet?.transactions || []));
    } catch (err: any) {
      setError(err?.message || "Failed to load notifications.");
    } finally {
      setLoading(false);
    }
  }

  function markRead(id: string) {
    setRead((prev) => {
      const next = prev.includes(id) ? prev : [...prev, id];
      localStorage.setItem(READ_KEY, JSON.stringify(next));
      return next;
    });
  }

  function markAllRead() {
    const ids = alerts.map((a) => a.id);
    localStorage.setItem(READ_KEY, JSON.stringify(ids));
    setRead(ids);
  }

  const unread = alerts.filter((a) => !read.includes(a.id)).length;

  return (
    <div style={{ fontFamily: font, maxWidth: 860 }}>
      <PageHeader
        title="Notifications"
        subtitle={
          unread > 0 ? `${unread} unread` : "You're all caught up"
        }
        right={
          alerts.length > 0 ? (
            <Btn
              onClick={markAllRead}
              variant="ghost"
              color={COLORS.g400}
              size="sm"
            >
              <Check size={13} /> Mark all read
            </Btn>
          ) : undefined
        }
      />

      {loading && <p style={{ color: COLORS.g500 }}>Loading…</p>}

      {!loading && error && (
        <EmptyState
          icon={<Bell size={40} />}
          title="Couldn't load notifications"
          hint={error}
          action={<Btn onClick={load}>Retry</Btn>}
        />
      )}

      {!loading && !error && alerts.length === 0 && (
        <EmptyState
          icon={<Bell size={40} />}
          title="No notifications yet"
          hint="Submit a report — AI results, verifications and coin rewards will appear here."
        />
      )}

      {!loading &&
        !error &&
        alerts.map((a) => {
          const meta = KIND_META[a.kind];
          const isRead = read.includes(a.id);
          return (
            <div
              key={a.id}
              onClick={() => markRead(a.id)}
              style={{
                display: "flex",
                gap: 12,
                padding: "12px 4px",
                borderBottom: `1px solid ${COLORS.g100}`,
                cursor: "pointer",
                opacity: isRead ? 0.6 : 1,
              }}
            >
              <div
                style={{
                  width: 38,
                  height: 38,
                  borderRadius: 10,
                  background: meta.color + "20",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: meta.color,
                  flexShrink: 0,
                }}
              >
                {meta.icon}
              </div>
              <div style={{ flex: 1 }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    gap: 8,
                  }}
                >
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: isRead ? 600 : 800,
                      color: COLORS.g900,
                    }}
                  >
                    {a.title}
                  </div>
                  <span
                    style={{
                      fontSize: 10,
                      color: COLORS.g400,
                      flexShrink: 0,
                    }}
                  >
                    {a.time}
                  </span>
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: COLORS.g500,
                    marginTop: 2,
                    lineHeight: 1.5,
                  }}
                >
                  {a.msg}
                </div>
              </div>
              {!isRead && (
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: COLORS.cGreen,
                    flexShrink: 0,
                    marginTop: 4,
                  }}
                />
              )}
            </div>
          );
        })}
    </div>
  );
}

function buildAlerts(
  reports: Report[],
  txs: CoinTransaction[]
): Alert[] {
  const out: Alert[] = [];
  const fmt = (iso: string) =>
    new Date(iso).toLocaleString(undefined, {
      day: "numeric",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });

  for (const tx of txs) {
    if (Number(tx.amount) <= 0 || !tx.created_at) continue;
    out.push({
      id: `tx-${tx.id}`,
      kind: "reward",
      title: "Green Coins Earned!",
      msg: `${tx.note || "Reward credited"} (+${tx.amount} coins)`,
      time: fmt(tx.created_at),
      ts: new Date(tx.created_at).getTime(),
    });
  }
  for (const r of reports) {
    const ts = new Date(r.updated_at).getTime();
    if (r.status === "COMPLETED") {
      out.push({
        id: `done-${r.id}`,
        kind: "verified",
        title: "Pickup Completed",
        msg: `Your report "${r.description || "waste report"}" was collected and verified.`,
        time: fmt(r.updated_at),
        ts,
      });
    } else if (r.prediction) {
      out.push({
        id: `ai-${r.id}`,
        kind: "ai",
        title: "AI Result Ready",
        msg: `Your report was classified as ${r.prediction.category_name || "waste"} (${Number(r.prediction.confidence ?? 0).toFixed(0)}% confidence).`,
        time: fmt(r.updated_at),
        ts,
      });
    }
  }
  return out.sort((a, b) => b.ts - a.ts).slice(0, 30);
}

export default Alerts;
