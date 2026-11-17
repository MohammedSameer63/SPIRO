import { useEffect, useState } from "react";
import { QrCode, ShieldCheck } from "lucide-react";
import { getCurrentUser } from "../../api/auth";
import { getHouseholdQr, type HouseholdQr } from "../../api/qr";
import { apiRequest } from "../../api/client";
import { Btn, Card, EmptyState, PageHeader } from "../../components/ui";
import { COLORS, font } from "../../theme";

type Household = {
  id: string;
  ward_id: string;
  house_number: string;
  street_name: string | null;
  address: string;
};

type Ward = { id: string; name: string; zone: string | null };

function MyQR() {
  const [qr, setQr] = useState<HouseholdQr | null>(null);
  const [household, setHousehold] = useState<Household | null>(null);
  const [wardName, setWardName] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    load();
  }, []);

  async function load() {
    try {
      setLoading(true);
      setError("");
      const me = await getCurrentUser();
      if (!me.household_id) {
        throw new Error("No household linked to this account.");
      }
      const [code, home] = await Promise.all([
        getHouseholdQr(me.household_id),
        apiRequest<Household>(`/households/${me.household_id}`, {
          method: "GET",
        }),
      ]);
      setQr(code);
      setHousehold(home);
      try {
        const wards = await apiRequest<Ward[]>("/wards", { method: "GET" });
        setWardName(wards.find((w) => w.id === home.ward_id)?.name || "");
      } catch {
        setWardName("");
      }
    } catch (err: any) {
      setError(err?.message || "Failed to load QR code.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{
        fontFamily: font,
        maxWidth: 560,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 16,
      }}
    >
      <div style={{ alignSelf: "flex-start", width: "100%" }}>
        <PageHeader
          title="My Household QR"
          subtitle="Workers scan this at pickup to verify collection"
        />
      </div>

      {loading && <p style={{ color: COLORS.g500 }}>Loading QR code…</p>}

      {!loading && error && (
        <EmptyState
          icon={<QrCode size={40} />}
          title="Couldn't load QR code"
          hint={error}
          action={<Btn onClick={load}>Retry</Btn>}
        />
      )}

      {!loading && !error && qr && (
        <>
          <div
            style={{
              background: "#fff",
              padding: 14,
              borderRadius: 12,
              boxShadow: "0 2px 12px rgba(0,0,0,0.15)",
            }}
          >
            <img
              src={qr.qrCode}
              alt="Household QR code"
              style={{ width: 220, height: 220, display: "block" }}
            />
          </div>
          <Card style={{ padding: 16, width: "100%" }}>
            {[
              ["House", household ? `${household.house_number}, ${household.street_name || ""}`.trim() : "—"],
              ["Ward", wardName || "—"],
              ["Address", household?.address || "—"],
              ["QR Security", "HMAC-SHA256 signed"],
              ["Status", "Active · Permanent"],
            ].map(([k, v], i, arr) => (
              <div
                key={k}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 12,
                  padding: "7px 0",
                  borderBottom:
                    i < arr.length - 1 ? `1px solid ${COLORS.g100}` : "none",
                  fontSize: 13,
                }}
              >
                <span style={{ color: COLORS.g400 }}>{k}</span>
                <span
                  style={{
                    fontWeight: 600,
                    color: COLORS.g900,
                    textAlign: "right",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 5,
                  }}
                >
                  {k === "QR Security" && (
                    <ShieldCheck size={14} color={COLORS.cGreen} />
                  )}
                  {v}
                </span>
              </div>
            ))}
          </Card>
        </>
      )}
    </div>
  );
}

export default MyQR;
