import { useEffect, useState } from "react";
import { ArrowLeft, Check, ScrollText } from "lucide-react";
import {
  getAdminReports,
  type AdminReport,
} from "../../api/admin";
import {
  Btn,
  Card,
  CoinTag,
  EmptyState,
  PageHeader,
  StatusPill,
  WastePill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

const FILTERS = ["All", "PENDING", "ACCEPTED", "IN_PROGRESS", "COMPLETED"] as const;

function Audit() {
  const [reports, setReports] = useState<AdminReport[]>([]);
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("All");
  const [selected, setSelected] = useState<AdminReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    load(filter === "All" ? undefined : filter);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter]);

  async function load(status?: string) {
    try {
      setLoading(true);
      setError("");
      const res = await getAdminReports(status);
      if (res.success) {
        setReports(res.data);
        setSelected((cur) =>
          cur ? (res.data.find((r) => r.id === cur.id) ?? null) : null
        );
      }
    } catch (err: any) {
      setError(err?.message || "Failed to load audit trail.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ fontFamily: font, maxWidth: 1120 }}>
      <PageHeader
        title="Full Audit Trail"
        subtitle="Every report traceable end-to-end"
      />

      <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              border: "none",
              borderRadius: 999,
              padding: "6px 14px",
              fontSize: 12,
              fontWeight: 700,
              cursor: "pointer",
              background: filter === f ? COLORS.aPurple : COLORS.g100,
              color: filter === f ? "#fff" : COLORS.g600,
            }}
          >
            {f === "All" ? "All" : f.charAt(0) + f.slice(1).toLowerCase()}
          </button>
        ))}
      </div>

      {loading && <p style={{ color: COLORS.g500 }}>Loading…</p>}

      {!loading && error && (
        <EmptyState
          icon={<ScrollText size={40} />}
          title="Couldn't load reports"
          hint={error}
          action={<Btn onClick={() => load()}>Retry</Btn>}
        />
      )}

      {!loading && !error && selected && (
        <DetailView
          report={selected}
          onBack={() => setSelected(null)}
        />
      )}

      {!loading && !error && !selected && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
            gap: 12,
          }}
        >
          {reports.map((r) => (
            <Card
              key={r.id}
              onClick={() => setSelected(r)}
              style={{ padding: 14 }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  marginBottom: 6,
                }}
              >
                <div
                  style={{ fontSize: 13, fontWeight: 700, color: COLORS.g900 }}
                >
                  {r.description || "Waste report"}
                </div>
                <StatusPill status={r.status} />
              </div>
              <div style={{ fontSize: 11, color: COLORS.g400, marginBottom: 8 }}>
                {new Date(r.created_at).toLocaleString()}
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                {r.prediction ? (
                  <WastePill category={r.prediction.category_name} />
                ) : (
                  <span style={{ fontSize: 11, color: COLORS.g400 }}>
                    No AI result
                  </span>
                )}
                {typeof r.coins_awarded === "number" &&
                  r.coins_awarded > 0 && (
                    <span style={{ marginLeft: "auto" }}>
                      <CoinTag amount={r.coins_awarded} />
                    </span>
                  )}
              </div>
            </Card>
          ))}
          {reports.length === 0 && (
            <div style={{ fontSize: 13, color: COLORS.g400 }}>
              No reports match this filter.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DetailView({
  report: r,
  onBack,
}: {
  report: AdminReport;
  onBack: () => void;
}) {
  const steps: { label: string; done: boolean }[] = [
    { label: "Citizen submitted report", done: true },
    {
      label: r.prediction
        ? `AI classified as ${r.prediction.category_name || "waste"} (${Number(r.prediction.confidence ?? 0).toFixed(0)}%)`
        : "AI analysis pending",
      done: !!r.prediction,
    },
    {
      label:
        r.status === "PENDING"
          ? "Awaiting worker assignment"
          : "Worker assigned / accepted",
      done: r.status !== "PENDING",
    },
    {
      label: r.pickup_validated
        ? "Household QR scanned at pickup"
        : "Pickup QR validation pending",
      done: !!r.pickup_validated,
    },
    {
      label:
        r.status === "COMPLETED"
          ? "Collection confirmed"
          : "Collection in progress",
      done: r.status === "COMPLETED",
    },
    {
      label:
        typeof r.coins_awarded === "number" && r.coins_awarded > 0
          ? `Green Coins credited (+${r.coins_awarded})`
          : "Green Coins pending completion",
      done: (r.coins_awarded || 0) > 0,
    },
  ];

  return (
    <div>
      <Btn
        onClick={onBack}
        variant="ghost"
        color={COLORS.g400}
        size="sm"
        style={{ marginBottom: 12 }}
      >
        <ArrowLeft size={13} /> All reports
      </Btn>
      <Card style={{ padding: 16, marginBottom: 12 }}>
        <div
          style={{
            fontSize: 16,
            fontWeight: 800,
            color: COLORS.g900,
            marginBottom: 12,
          }}
        >
          {r.description || "Waste report"}
        </div>
        {[
          ["Report ID", r.id.slice(0, 8) + "…"],
          ["Status", r.status],
          ["AI Category", r.prediction?.category_name || "—"],
          [
            "Confidence",
            r.prediction
              ? `${Number(r.prediction.confidence ?? 0).toFixed(1)}%`
              : "—",
          ],
          ["Model", r.prediction?.model_used || "—"],
          ["QR Validated", r.pickup_validated ? "Yes" : "No"],
          ["Coins", String(r.coins_awarded ?? 0)],
          ["Filed", new Date(r.created_at).toLocaleString()],
          ["Updated", new Date(r.updated_at).toLocaleString()],
        ].map(([k, v]) => (
          <div
            key={k}
            style={{
              display: "flex",
              justifyContent: "space-between",
              padding: "6px 0",
              borderBottom: `1px solid ${COLORS.g100}`,
              fontSize: 13,
            }}
          >
            <span style={{ color: COLORS.g500 }}>{k}</span>
            <span style={{ fontWeight: 600, color: COLORS.g900 }}>{v}</span>
          </div>
        ))}
      </Card>
      <div style={{ display: "flex", flexDirection: "column" }}>
        {steps.map((s, i) => (
          <div
            key={s.label}
            style={{ display: "flex", gap: 10, alignItems: "flex-start" }}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              <div
                style={{
                  width: 20,
                  height: 20,
                  borderRadius: "50%",
                  background: s.done ? COLORS.cGreen : COLORS.g300,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#fff",
                  flexShrink: 0,
                }}
              >
                {s.done && <Check size={11} />}
              </div>
              {i < steps.length - 1 && (
                <div style={{ width: 2, height: 20, background: COLORS.g200 }} />
              )}
            </div>
            <div
              style={{
                fontSize: 12,
                color: s.done ? COLORS.g700 : COLORS.g400,
                paddingTop: 2,
                paddingBottom: 18,
              }}
            >
              {s.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Audit;
