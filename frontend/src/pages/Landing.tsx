import { useNavigate } from "react-router-dom";
import {
  Bell,
  Brain,
  Camera,
  Coins,
  Landmark,
  QrCode,
  Recycle,
  ShieldCheck,
  Truck,
  Zap,
} from "lucide-react";
import { Btn } from "../components/ui";
import { COLORS, font } from "../theme";

const STATS = [
  { v: "34", l: "Waste Classes", icon: <Recycle size={18} /> },
  { v: "<300ms", l: "AI Inference", icon: <Zap size={18} /> },
  { v: "QR", l: "Pickup Validation", icon: <QrCode size={18} /> },
  { v: "YOLOv11", l: "Vision AI", icon: <Brain size={18} /> },
  { v: "HMAC", l: "QR Security", icon: <ShieldCheck size={18} /> },
  { v: "Live", l: "Real AI", icon: <Bell size={18} /> },
];

const FLOW = [
  ["Scan or upload a waste photo", <Camera size={16} key="c" />],
  ["AI classifies it with the YOLOv11 pipeline", <Brain size={16} key="b" />],
  ["Disposal guidance plus coin estimate", <Recycle size={16} key="r" />],
  ["Worker accepts, scans QR at pickup", <Truck size={16} key="t" />],
  ["Collection confirmed on completion", <ShieldCheck size={16} key="s" />],
  ["Green Coins credited to citizen wallet", <Coins size={16} key="g" />],
  ["Admin audits the full traceable trail", <Landmark size={16} key="a" />],
] as const;

function Landing() {
  const navigate = useNavigate();
  return (
    <div
      style={{
        fontFamily: font,
        minHeight: "100vh",
        background: COLORS.g50,
      }}
    >
      <div
        style={{
          background: "linear-gradient(135deg,#0f2027,#203a43,#2c5364)",
          padding: "56px 24px",
          textAlign: "center",
        }}
      >
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            background: "rgba(255,255,255,.1)",
            borderRadius: 20,
            padding: "6px 16px",
            marginBottom: 20,
          }}
        >
          <div
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: COLORS.cGreen,
            }}
          />
          <span
            style={{ fontSize: 12, color: "rgba(255,255,255,.8)", fontWeight: 600 }}
          >
            YOLOv11 · ONNX Runtime · 34 Waste Classes
          </span>
        </div>
        <div
          style={{
            fontSize: 34,
            fontWeight: 900,
            color: "#fff",
            lineHeight: 1.1,
            marginBottom: 14,
            letterSpacing: -1,
          }}
        >
          Smart Waste Collection
          <br />
          <span style={{ color: COLORS.cGreen }}>Powered by SPIRO AI</span>
        </div>
        <div
          style={{
            fontSize: 14,
            color: "rgba(255,255,255,.7)",
            lineHeight: 1.7,
            marginBottom: 24,
            maxWidth: 480,
            margin: "0 auto 24px",
          }}
        >
          Report waste with a photo — SPIRO AI classifies it instantly,
          guides disposal, then drives collection and rewards end-to-end.
        </div>
        <div
          style={{
            display: "flex",
            gap: 10,
            maxWidth: 420,
            margin: "0 auto",
            flexWrap: "wrap",
            justifyContent: "center",
          }}
        >
          <Btn onClick={() => navigate("/login")} size="lg">
            Open App
          </Btn>
          <Btn
            onClick={() => navigate("/register")}
            variant="ghost"
            color="#fff"
            size="lg"
          >
            Create Account
          </Btn>
        </div>
      </div>

      <div
        style={{
          background: COLORS.cGreen,
          padding: "18px 16px",
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
          gap: 12,
          maxWidth: 1120,
          margin: "0 auto",
        }}
      >
        {STATS.map((s) => (
          <div key={s.l} style={{ textAlign: "center", color: "#fff" }}>
            <div>{s.icon}</div>
            <div style={{ fontSize: 20, fontWeight: 900 }}>{s.v}</div>
            <div style={{ fontSize: 10, color: "rgba(255,255,255,.7)" }}>
              {s.l}
            </div>
          </div>
        ))}
      </div>

      <div
        style={{
          padding: 24,
          display: "flex",
          flexDirection: "column",
          gap: 10,
          maxWidth: 720,
          margin: "0 auto",
        }}
      >
        <div
          style={{
            fontSize: 16,
            fontWeight: 800,
            color: COLORS.g900,
            textAlign: "center",
            marginBottom: 4,
          }}
        >
          How SPIRO Works
        </div>
        {FLOW.map(([text, icon], i) => (
          <div
            key={i}
            style={{
              display: "flex",
              gap: 10,
              alignItems: "center",
              background: COLORS.white,
              borderRadius: 10,
              padding: "10px 14px",
              border: `1px solid ${COLORS.g200}`,
            }}
          >
            <div
              style={{
                width: 24,
                height: 24,
                borderRadius: "50%",
                background: COLORS.cGreen,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#fff",
                fontSize: 11,
                fontWeight: 700,
                flexShrink: 0,
              }}
            >
              {i + 1}
            </div>
            <span style={{ color: COLORS.g500 }}>{icon}</span>
            <span style={{ fontSize: 13, color: COLORS.g700 }}>{text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Landing;
