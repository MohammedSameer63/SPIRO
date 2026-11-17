// SPIRO design tokens — palette adopted from the Major_Artifact demo UI,
// applied across the whole app for one consistent look.

export const COLORS = {
  cGreen: "#16a34a",
  cGreenL: "#dcfce7",
  cGreenD: "#14532d",
  cEmerald: "#059669",
  dBlue: "#1d4ed8",
  dBlueL: "#dbeafe",
  dBlueD: "#1e3a8a",
  aPurple: "#7c3aed",
  aPurpleL: "#f3e8ff",
  red: "#dc2626",
  redL: "#fee2e2",
  amber: "#d97706",
  amberL: "#fef3c7",
  sky: "#0284c7",
  skyL: "#e0f2fe",
  g50: "#f9fafb",
  g100: "#f3f4f6",
  g200: "#e5e7eb",
  g300: "#d1d5db",
  g400: "#9ca3af",
  g500: "#6b7280",
  g600: "#4b5563",
  g700: "#374151",
  g800: "#1f2937",
  g900: "#111827",
  white: "#ffffff",
  coin: "#f59e0b",
} as const;

export const font =
  "-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif";

// Waste streams as stored in waste_categories (canonical names).
export type WasteStreamKey =
  | "recoverable"
  | "organic"
  | "non-recoverable"
  | "hazardous"
  | "sanitary"
  | "e-waste"
  | "reject";

export const WASTE_STREAMS: Record<
  WasteStreamKey,
  { label: string; color: string; bg: string; bin: string }
> = {
  recoverable: {
    label: "Recoverable",
    color: COLORS.dBlue,
    bg: COLORS.dBlueL,
    bin: "Blue bin",
  },
  organic: {
    label: "Organic",
    color: COLORS.cGreen,
    bg: COLORS.cGreenL,
    bin: "Brown bin",
  },
  "non-recoverable": {
    label: "Non-Recoverable",
    color: COLORS.g500,
    bg: COLORS.g100,
    bin: "Grey bin",
  },
  hazardous: {
    label: "Hazardous",
    color: COLORS.red,
    bg: COLORS.redL,
    bin: "Red bin",
  },
  sanitary: {
    label: "Sanitary",
    color: COLORS.aPurple,
    bg: COLORS.aPurpleL,
    bin: "Purple bin",
  },
  "e-waste": {
    label: "E-Waste",
    color: COLORS.amber,
    bg: COLORS.amberL,
    bin: "Orange bin",
  },
  reject: {
    label: "Reject",
    color: "#78350f",
    bg: COLORS.amberL,
    bin: "Black bin",
  },
};

/** Normalize any category/stream spelling to a stream key. */
export function toStreamKey(
  name: string | null | undefined
): WasteStreamKey {
  const n = (name || "")
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (n.includes("recoverable") && !n.includes("non")) return "recoverable";
  if (n.includes("non") && n.includes("recover")) return "non-recoverable";
  if (n.includes("organic") || n.includes("food") || n.includes("compost"))
    return "organic";
  if (n.includes("hazard")) return "hazardous";
  if (
    n.includes("sanitar") ||
    n.includes("medical") ||
    n.includes("hygiene")
  )
    return "sanitary";
  if (
    n.includes("e waste") ||
    n.includes("ewaste") ||
    n.includes("electronic") ||
    n.includes("battery")
  )
    return "e-waste";
  return "reject";
}

export type ReportStatusKey =
  | "PENDING"
  | "ACCEPTED"
  | "IN_PROGRESS"
  | "COMPLETED";

export const STATUS_META: Record<
  ReportStatusKey,
  { label: string; color: string; bg: string }
> = {
  PENDING: { label: "Pending", color: COLORS.amber, bg: COLORS.amberL },
  ACCEPTED: { label: "Accepted", color: COLORS.dBlue, bg: COLORS.dBlueL },
  IN_PROGRESS: {
    label: "In Progress",
    color: COLORS.sky,
    bg: COLORS.skyL,
  },
  COMPLETED: {
    label: "Completed",
    color: COLORS.cGreen,
    bg: COLORS.cGreenL,
  },
};

// Disposal guidance per stream (client-side copy of the ML guidance).
export const DISPOSAL_GUIDE: Record<WasteStreamKey, string[]> = {
  recoverable: [
    "Make sure items are empty, clean and dry",
    "Keep caps on bottles — facilities sort them together",
    "Place in the Blue (Recoverable) bin — do not crush flat",
  ],
  organic: [
    "Place food and garden waste directly in the Brown bin",
    "No washing or sorting needed for clean organics",
    "Remove any plastic bags or packaging mixed in",
  ],
  "non-recoverable": [
    "Place in the Grey (general waste) bin",
    "Not suitable for recycling or composting",
  ],
  hazardous: [
    "Do NOT place in any regular bin",
    "Take to a designated hazardous-waste drop-off point",
    "Store cool and dry until collection (batteries = fire risk)",
  ],
  sanitary: [
    "Seal in a bag before binning",
    "Do NOT recycle sanitary or medical waste",
  ],
  "e-waste": [
    "Do not place in regular bins",
    "Hand over during e-waste collection or at a drop-off point",
    "Wipe personal data from devices first",
  ],
  reject: [
    "A worker will verify the category during collection",
    "Keep it separate until then",
  ],
};

export const DO_NOT_MIX: Record<WasteStreamKey, string[]> = {
  recoverable: ["Organic waste", "Hazardous chemicals", "Sanitary waste"],
  organic: ["Recoverable waste", "Hazardous waste", "Sanitary waste"],
  "non-recoverable": ["Hazardous waste", "Sanitary waste"],
  hazardous: ["All other waste streams"],
  sanitary: ["Recoverable waste", "Organic waste"],
  "e-waste": ["Organic", "Recoverable", "Hazardous"],
  reject: ["Hazardous waste", "Sanitary waste"],
};
