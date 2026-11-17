import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { MapPin, Navigation, Route as RouteIcon } from "lucide-react";
import { getWorkerQueue } from "../../api/reports";
import type { Report } from "../../api/reports";
import {
  Btn,
  Card,
  EmptyState,
  PageHeader,
  WastePill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

function RoutePage() {
  const navigate = useNavigate();
  const [stops, setStops] = useState<Report[]>([]);
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
      const res = await getWorkerQueue();
      if (res.success) {
        setStops(
          [...res.data].sort(
            (a, b) =>
              new Date(a.created_at).getTime() -
              new Date(b.created_at).getTime()
          )
        );
      }
    } catch (err: any) {
      setError(err?.message || "Failed to load route.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ fontFamily: font, maxWidth: 860 }}>
      <PageHeader
        title="Today's Route"
        subtitle={
          stops.length > 0
            ? `${stops.length} stop${stops.length === 1 ? "" : "s"} · oldest first`
            : undefined
        }
        right={
          <Btn size="sm" variant="ghost" onClick={load}>
            Refresh
          </Btn>
        }
      />

      {loading && <p style={{ color: COLORS.g500 }}>Loading route…</p>}

      {!loading && error && (
        <EmptyState
          icon={<RouteIcon size={40} />}
          title="Couldn't load route"
          hint={error}
          action={<Btn onClick={load}>Retry</Btn>}
        />
      )}

      {!loading && !error && stops.length === 0 && (
        <EmptyState
          icon={<Navigation size={40} />}
          title="No stops today"
          hint="Accept reports from the queue and they will appear here in order."
          action={<Btn onClick={() => navigate("/worker/queue")}>Open Queue</Btn>}
        />
      )}

      {!loading && !error && stops.length > 0 && (
        <>
          <Card
            style={{
              padding: 14,
              marginBottom: 12,
              background: "linear-gradient(135deg,#1e3a8a,#1d4ed8)",
            }}
          >
            <div style={{ color: "rgba(255,255,255,.7)", fontSize: 11, marginBottom: 6 }}>
              Route Summary
            </div>
            <div style={{ display: "flex", gap: 24 }}>
              {[
                { v: `${stops.length}`, l: "Stops" },
                {
                  v: new Set(stops.map((s) => s.address || "?")).size.toString(),
                  l: "Locations",
                },
              ].map((s) => (
                <div key={s.l}>
                  <div style={{ fontSize: 20, fontWeight: 900, color: "#fff" }}>
                    {s.v}
                  </div>
                  <div style={{ fontSize: 10, color: "rgba(255,255,255,.6)" }}>
                    {s.l}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {stops.map((s, i) => (
            <Card key={s.id} style={{ padding: 14, marginBottom: 10 }}>
              <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                <div
                  style={{
                    width: 30,
                    height: 30,
                    borderRadius: "50%",
                    background: COLORS.dBlue,
                    color: "#fff",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 800,
                    flexShrink: 0,
                  }}
                >
                  {i + 1}
                </div>
                <div style={{ flex: 1 }}>
                  <div
                    style={{ fontSize: 13, fontWeight: 700, color: COLORS.g900 }}
                  >
                    {s.description || "Waste report"}
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: COLORS.g500,
                      display: "flex",
                      alignItems: "center",
                      gap: 4,
                      marginTop: 2,
                    }}
                  >
                    <MapPin size={11} /> {s.address || "No address"}
                  </div>
                  <div style={{ marginTop: 6 }}>
                    {s.prediction ? (
                      <WastePill category={s.prediction.category_name} />
                    ) : (
                      <span style={{ fontSize: 11, color: COLORS.g400 }}>
                        AI analyzing…
                      </span>
                    )}
                  </div>
                </div>
                <Btn size="sm" onClick={() => navigate(`/worker/verify/${s.id}`)}>
                  Verify
                </Btn>
              </div>
            </Card>
          ))}
        </>
      )}
    </div>
  );
}

export default RoutePage;
