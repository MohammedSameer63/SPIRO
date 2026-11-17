import type { ReactNode } from "react";
import {
  BatteryCharging,
  Check,
  CircleHelp,
  ClipboardCheck,
  Clock,
  Coins,
  Leaf,
  LoaderCircle,
  Recycle,
  Syringe,
  Trash2,
  TriangleAlert,
  Truck,
} from "lucide-react";
import {
  COLORS,
  STATUS_META,
  WASTE_STREAMS,
  toStreamKey,
  type ReportStatusKey,
  type WasteStreamKey,
} from "../theme";

// ---------- atoms (artifact design language) ----------

export function Pill({
  children,
  color = COLORS.g600,
  bg = COLORS.g100,
}: {
  children: ReactNode;
  color?: string;
  bg?: string;
}) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        background: bg,
        color,
        borderRadius: 999,
        padding: "2px 8px",
        fontSize: 11,
        fontWeight: 600,
        whiteSpace: "nowrap",
        lineHeight: 1.5,
      }}
    >
      {children}
    </span>
  );
}

const STREAM_ICONS: Record<WasteStreamKey, ReactNode> = {
  recoverable: <Recycle size={12} />,
  organic: <Leaf size={12} />,
  "non-recoverable": <Trash2 size={12} />,
  hazardous: <TriangleAlert size={12} />,
  sanitary: <Syringe size={12} />,
  "e-waste": <BatteryCharging size={12} />,
  reject: <CircleHelp size={12} />,
};

export function WastePill({
  category,
}: {
  category: string | null | undefined;
}) {
  const key = toStreamKey(category);
  const w = WASTE_STREAMS[key];
  return (
    <Pill color={w.color} bg={w.bg}>
      {STREAM_ICONS[key]} {w.label}
    </Pill>
  );
}

const STATUS_ICONS: Record<ReportStatusKey, ReactNode> = {
  PENDING: <Clock size={12} />,
  ACCEPTED: <ClipboardCheck size={12} />,
  IN_PROGRESS: <Truck size={12} />,
  COMPLETED: <Check size={12} />,
};

export function StatusPill({ status }: { status: string }) {
  const meta =
    STATUS_META[status as ReportStatusKey] ||
    ({ label: status, color: COLORS.g500, bg: COLORS.g100 } as const);
  const icon =
    STATUS_ICONS[status as ReportStatusKey] || null;
  return (
    <Pill color={meta.color} bg={meta.bg}>
      {icon} {meta.label}
    </Pill>
  );
}

export function Card({
  children,
  style = {},
  onClick,
}: {
  children: ReactNode;
  style?: React.CSSProperties;
  onClick?: () => void;
}) {
  return (
    <div
      onClick={onClick}
      style={{
        background: COLORS.white,
        borderRadius: 16,
        border: `1px solid ${COLORS.g200}`,
        boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
        cursor: onClick ? "pointer" : "default",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

type BtnVariant = "solid" | "ghost" | "subtle";

export function Btn({
  children,
  onClick,
  color = COLORS.cGreen,
  variant = "solid",
  size = "md",
  disabled,
  style = {},
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  color?: string;
  variant?: BtnVariant;
  size?: "sm" | "md" | "lg";
  disabled?: boolean;
  style?: React.CSSProperties;
  type?: "button" | "submit";
}) {
  const base: React.CSSProperties = {
    border: "none",
    borderRadius: 10,
    cursor: disabled ? "not-allowed" : "pointer",
    fontWeight: 600,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    opacity: disabled ? 0.5 : 1,
    fontSize: size === "sm" ? 12 : size === "lg" ? 16 : 14,
    padding:
      size === "sm"
        ? "6px 12px"
        : size === "lg"
          ? "13px 28px"
          : "9px 18px",
  };
  const variants: Record<BtnVariant, React.CSSProperties> = {
    solid: { background: color, color: "#fff" },
    ghost: {
      background: "transparent",
      color,
      border: `1.5px solid ${color}`,
    },
    subtle: { background: color + "18", color, border: "none" },
  };
  return (
    <button
      type={type}
      onClick={disabled ? undefined : onClick}
      style={{ ...base, ...variants[variant], ...style }}
    >
      {children}
    </button>
  );
}

export function Stat({
  label,
  value,
  icon,
  color = COLORS.cGreen,
}: {
  label: string;
  value: ReactNode;
  icon: ReactNode;
  color?: string;
}) {
  return (
    <Card style={{ padding: "14px 16px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
        }}
      >
        <div>
          <div
            style={{
              fontSize: 10,
              color: COLORS.g400,
              fontWeight: 700,
              letterSpacing: 0.8,
              textTransform: "uppercase",
              marginBottom: 5,
            }}
          >
            {label}
          </div>
          <div
            style={{
              fontSize: 24,
              fontWeight: 900,
              color: COLORS.g900,
              lineHeight: 1,
            }}
          >
            {value}
          </div>
        </div>
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 9,
            background: color + "20",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color,
          }}
        >
          {icon}
        </div>
      </div>
    </Card>
  );
}

export function Bar({
  v,
  max = 100,
  color = COLORS.cGreen,
  h = 6,
}: {
  v: number;
  max?: number;
  color?: string;
  h?: number;
}) {
  return (
    <div
      style={{
        height: h,
        background: COLORS.g100,
        borderRadius: h,
        overflow: "hidden",
        flex: 1,
      }}
    >
      <div
        style={{
          height: "100%",
          width: `${Math.min(100, (v / max) * 100)}%`,
          background: color,
          borderRadius: h,
          transition: "width 1s ease",
        }}
      />
    </div>
  );
}

export function Spinner({ color = COLORS.cGreen }: { color?: string }) {
  return (
    <>
      <style>{`@keyframes spiro-spin{to{transform:rotate(360deg);}}`}</style>
      <LoaderCircle
        size={16}
        color={color}
        style={{ animation: "spiro-spin 0.7s linear infinite" }}
      />
    </>
  );
}

export function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        gap: 12,
        marginBottom: 18,
      }}
    >
      <div>
        <div
          style={{ fontSize: 22, fontWeight: 800, color: COLORS.g900 }}
        >
          {title}
        </div>
        {subtitle && (
          <div
            style={{ fontSize: 12, color: COLORS.g400, marginTop: 4 }}
          >
            {subtitle}
          </div>
        )}
      </div>
      {right}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  hint,
  action,
}: {
  icon: ReactNode;
  title: string;
  hint?: string;
  action?: ReactNode;
}) {
  return (
    <Card style={{ padding: 36, textAlign: "center" }}>
      <div
        style={{ color: COLORS.g300, marginBottom: 12 }}
      >
        {icon}
      </div>
      <div
        style={{
          fontSize: 16,
          fontWeight: 700,
          color: COLORS.g900,
        }}
      >
        {title}
      </div>
      {hint && (
        <div
          style={{
            fontSize: 13,
            color: COLORS.g500,
            marginTop: 6,
          }}
        >
          {hint}
        </div>
      )}
      {action && (
        <div style={{ marginTop: 14 }}>{action}</div>
      )}
    </Card>
  );
}

export function CoinTag({ amount }: { amount: number }) {
  return (
    <span
      style={{
        fontSize: 12,
        fontWeight: 700,
        color: COLORS.coin,
        display: "inline-flex",
        alignItems: "center",
        gap: 3,
      }}
    >
      <Coins size={13} />+{amount}
    </span>
  );
}
