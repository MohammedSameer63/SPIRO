import { useEffect, useState } from "react";
import { ArrowDownLeft, ArrowUpRight, Coins, Info, Wallet as WalletIcon } from "lucide-react";
import { getWallet, type CoinWallet } from "../../api/coins";
import { Btn, Card, EmptyState, PageHeader } from "../../components/ui";
import { COLORS, font } from "../../theme";

function Wallet() {
  const [wallet, setWallet] = useState<CoinWallet | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    load();
  }, []);

  async function load() {
    try {
      setLoading(true);
      setError("");
      setWallet(await getWallet());
    } catch (err: any) {
      setError(err?.message || "Failed to load wallet.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{ fontFamily: font, maxWidth: 860, display: "flex", flexDirection: "column", gap: 14 }}
    >
      <PageHeader
        title="Green Coin Wallet"
        subtitle="Earned for verified collections"
      />

      {loading && <p style={{ color: COLORS.g500 }}>Loading wallet…</p>}

      {!loading && error && (
        <EmptyState
          icon={<WalletIcon size={40} />}
          title="Couldn't load wallet"
          hint={error}
          action={<Btn onClick={load}>Retry</Btn>}
        />
      )}

      {!loading && !error && wallet && (
        <>
          <div
            style={{
              background: `linear-gradient(135deg,${COLORS.cGreen},${COLORS.cEmerald})`,
              borderRadius: 16,
              padding: 24,
            }}
          >
            <div
              style={{
                color: "rgba(255,255,255,.75)",
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: 1,
                textTransform: "uppercase",
                marginBottom: 6,
              }}
            >
              Green Coin Balance
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Coins size={44} color="#fff" />
              <span
                style={{ fontSize: 52, fontWeight: 900, color: "#fff", lineHeight: 1 }}
              >
                {wallet.balance}
              </span>
            </div>
            <div
              style={{
                display: "flex",
                gap: 10,
                marginTop: 16,
              }}
            >
              <div
                style={{
                  flex: 1,
                  background: "rgba(255,255,255,.2)",
                  borderRadius: 10,
                  padding: "10px 12px",
                }}
              >
                <div style={{ fontSize: 18, fontWeight: 800, color: "#fff" }}>
                  {wallet.total_earned}
                </div>
                <div style={{ fontSize: 10, color: "rgba(255,255,255,.7)" }}>
                  Lifetime earned
                </div>
              </div>
              <div
                style={{
                  flex: 1,
                  background: "rgba(255,255,255,.2)",
                  borderRadius: 10,
                  padding: "10px 12px",
                }}
              >
                <div style={{ fontSize: 18, fontWeight: 800, color: "#fff" }}>
                  {wallet.transactions.length}
                </div>
                <div style={{ fontSize: 10, color: "rgba(255,255,255,.7)" }}>
                  Rewards received
                </div>
              </div>
            </div>
          </div>

          <Card style={{ padding: 16 }}>
            <div
              style={{ display: "flex", gap: 8, alignItems: "flex-start" }}
            >
              <Info size={16} color={COLORS.dBlue} style={{ flexShrink: 0, marginTop: 1 }} />
              <div style={{ fontSize: 12, color: COLORS.g600, lineHeight: 1.6 }}>
                Coins are credited automatically when a worker marks your
                report <strong>Completed</strong>. Pickups verified with
                your Household QR earn a <strong>bonus</strong>.
              </div>
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
              Transactions
            </div>
            {wallet.transactions.length === 0 ? (
              <div style={{ fontSize: 13, color: COLORS.g400 }}>
                No transactions yet — complete your first report to earn
                coins.
              </div>
            ) : (
              wallet.transactions.map((tx, i) => {
                const earned = Number(tx.amount) >= 0;
                return (
                  <div
                    key={tx.id}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "9px 0",
                      borderBottom:
                        i < wallet.transactions.length - 1
                          ? `1px solid ${COLORS.g100}`
                          : "none",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                      }}
                    >
                      <div
                        style={{
                          width: 34,
                          height: 34,
                          borderRadius: "50%",
                          background: (earned ? COLORS.cGreen : COLORS.red) + "20",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        {earned ? (
                          <ArrowDownLeft size={15} color={COLORS.cGreen} />
                        ) : (
                          <ArrowUpRight size={15} color={COLORS.red} />
                        )}
                      </div>
                      <div>
                        <div
                          style={{
                            fontSize: 12,
                            fontWeight: 600,
                            color: COLORS.g900,
                          }}
                        >
                          {tx.note || tx.reason}
                        </div>
                        <div style={{ fontSize: 10, color: COLORS.g400 }}>
                          {tx.created_at
                            ? new Date(tx.created_at).toLocaleString()
                            : ""}
                        </div>
                      </div>
                    </div>
                    <span
                      style={{
                        fontSize: 14,
                        fontWeight: 700,
                        color: earned ? COLORS.cGreen : COLORS.red,
                      }}
                    >
                      {earned ? "+" : ""}
                      {tx.amount}
                    </span>
                  </div>
                );
              })
            )}
          </Card>
        </>
      )}
    </div>
  );
}

export default Wallet;
