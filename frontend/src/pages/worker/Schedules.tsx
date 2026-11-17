import { useEffect, useState } from "react";
import { CalendarDays, Clock } from "lucide-react";
import { getWorkerSchedules } from "../../api/reports";
import type { WorkerSchedule } from "../../api/reports";
import {
  Btn,
  Card,
  EmptyState,
  PageHeader,
  WastePill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

function Schedules() {
  const [schedules, setSchedules] =
    useState<WorkerSchedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadSchedules();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadSchedules() {
    try {
      setLoading(true);
      setError("");
      const response = await getWorkerSchedules();
      if (response.success) setSchedules(response.data);
    } catch (err: any) {
      setError(err?.message || "Failed to load collection schedules.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ fontFamily: font, maxWidth: 1120 }}>
      <PageHeader
        title="Collection Schedules"
        subtitle="Pickup timetable for your assigned wards"
        right={
          <Btn size="sm" variant="ghost" onClick={loadSchedules}>
            Refresh
          </Btn>
        }
      />

      {loading && <p style={{ color: COLORS.g500 }}>Loading schedules…</p>}

      {!loading && error && (
        <EmptyState
          icon={<CalendarDays size={40} />}
          title="Couldn't load schedules"
          hint={error}
          action={<Btn onClick={loadSchedules}>Retry</Btn>}
        />
      )}

      {!loading && !error && schedules.length === 0 && (
        <EmptyState
          icon={<CalendarDays size={40} />}
          title="No schedules assigned"
          hint="There are currently no collection schedules for your assigned wards."
        />
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
          gap: 12,
        }}
      >
        {schedules.map((s) => (
          <Card key={s.id} style={{ padding: 14 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                gap: 8,
                marginBottom: 10,
              }}
            >
              <div>
                <div
                  style={{ fontSize: 15, fontWeight: 800, color: COLORS.g900 }}
                >
                  {s.ward}
                </div>
                <div
                  style={{
                    fontSize: 11,
                    color: COLORS.g400,
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    marginTop: 2,
                  }}
                >
                  <Clock size={11} />
                  {s.day} · {s.startTime} – {s.endTime}
                </div>
              </div>
              <WastePill category={s.category} />
            </div>
            <div style={{ fontSize: 12, color: COLORS.g600 }}>
              {s.category} collection window
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

export default Schedules;
