import { useEffect, useState } from "react";
import { Brain, Check, Cpu } from "lucide-react";
import { getMlStatus, type MlStatus } from "../../api/admin";
import {
  Btn,
  Card,
  EmptyState,
  PageHeader,
  Pill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

function ML() {
  const [data, setData] = useState<MlStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    load();
  }, []);

  async function load() {
    try {
      setLoading(true);
      setError("");
      const res = await getMlStatus();
      if (res.success) setData(res.data);
    } catch (err: any) {
      setError(err?.message || "Failed to load ML status.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{
        fontFamily: font,
        maxWidth: 860,
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      <PageHeader
        title="AI/ML Model Registry"
        subtitle="Live inference state — no mock numbers"
        right={
          <Btn size="sm" variant="ghost" onClick={load}>
            Refresh
          </Btn>
        }
      />

      {loading && <p style={{ color: COLORS.g500 }}>Loading…</p>}

      {!loading && error && (
        <EmptyState
          icon={<Brain size={40} />}
          title="Couldn't load ML status"
          hint={error}
          action={<Btn onClick={load}>Retry</Btn>}
        />
      )}

      {!loading && !error && data && (
        <>
          <Card
            style={{ padding: 16, borderLeft: `4px solid ${COLORS.aPurple}` }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                marginBottom: 8,
              }}
            >
              <div>
                <div
                  style={{ fontSize: 15, fontWeight: 800, color: COLORS.g900 }}
                >
                  {data.architecture}
                </div>
                <div
                  style={{
                    fontSize: 11,
                    color: COLORS.g500,
                    fontFamily: "monospace",
                    marginTop: 2,
                  }}
                >
                  {data.model_file}
                </div>
              </div>
              <Pill
                color={data.enabled ? COLORS.cGreen : COLORS.g500}
                bg={data.enabled ? COLORS.cGreenL : COLORS.g100}
              >
                {data.enabled ? (
                  <>
                    <Check size={11} /> Enabled
                  </>
                ) : (
                  "Disabled"
                )}
              </Pill>
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
                gap: 8,
                marginTop: 10,
              }}
            >
              {[
                ["Mode", data.mock_mode ? "Mock (no model)" : "Live model"],
                ["Classes", String(data.num_classes)],
                ["Input", `${data.input_size}×${data.input_size}`],
                [
                  "Threshold",
                  `${Math.round(data.confidence_threshold * 100)}%`,
                ],
                ["Loaded", data.initialized ? "Yes" : "Not yet"],
              ].map(([k, v]) => (
                <div
                  key={k}
                  style={{
                    background: COLORS.g50,
                    borderRadius: 8,
                    padding: "8px 10px",
                    textAlign: "center",
                  }}
                >
                  <div style={{ fontSize: 10, color: COLORS.g400 }}>{k}</div>
                  <div
                    style={{
                      fontSize: 14,
                      fontWeight: 800,
                      color: COLORS.g900,
                    }}
                  >
                    {v}
                  </div>
                </div>
              ))}
            </div>
            <div
              style={{
                background: COLORS.cGreenL,
                borderRadius: 8,
                padding: 8,
                textAlign: "center",
                fontSize: 12,
                color: COLORS.cGreen,
                fontWeight: 600,
                marginTop: 10,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 6,
              }}
            >
              <Cpu size={14} />
              {data.mock_mode
                ? "Mock mode — predictions are simulated"
                : "Serving all inference requests"}
            </div>
          </Card>

          <Card style={{ padding: 16 }}>
            <div
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: COLORS.g900,
                marginBottom: 12,
              }}
            >
              Inference Pipeline
            </div>
            {data.pipeline_stages.map((s, i, arr) => (
              <div
                key={s}
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
                      width: 22,
                      height: 22,
                      borderRadius: "50%",
                      background: COLORS.aPurple,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "#fff",
                      fontSize: 10,
                      fontWeight: 700,
                    }}
                  >
                    {i + 1}
                  </div>
                  {i < arr.length - 1 && (
                    <div
                      style={{ width: 2, height: 18, background: COLORS.g200 }}
                    />
                  )}
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: COLORS.g700,
                    paddingTop: 3,
                    paddingBottom: 16,
                  }}
                >
                  {s}
                </div>
              </div>
            ))}
          </Card>
        </>
      )}
    </div>
  );
}

export default ML;
