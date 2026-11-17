import { useEffect, useId, useRef, useState } from "react";
import { Html5Qrcode } from "html5-qrcode";

interface QrScannerProps {
  onScan: (payload: string) => void;
  disabled?: boolean;
}

/*
 * Camera-only household QR scanner using html5-qrcode.
 *
 * FIXES for mobile webcam:
 * - The <div id={elementId}> is ALWAYS mounted (hidden when closed).
 *   html5-qrcode does document.getElementById(id) at start(); if the
 *   div is conditionally mounted, it throws "element not found" and
 *   the scanner never opens.
 * - Camera is opened via getUserMedia FIRST so the browser permission
 *   dialog appears and enumerateDevices() populates on mobile Chrome.
 * - Falls back from deviceId -> facingMode if the exact device id is
 *   rejected by the browser.
 * - Errors are surfaced with plain-language messages instead of the
 *   library's generic "unknown error".
 */
export function QrScanner({
  onScan,
  disabled = false,
}: QrScannerProps) {
  const elementId = `qr-reader-${useId().replace(/:/g, "")}`;
  const scannerRef = useRef<Html5Qrcode | null>(null);
  const onScanRef = useRef(onScan);
  onScanRef.current = onScan;

  const [open, setOpen] = useState(false);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);

  // Tear down camera when component unmounts
  useEffect(() => {
    return () => stopCamera();
  }, []);

  async function startScan() {
    setError("");
    setStarting(true);
    try {
      // Step 1: request camera permission so enumerateDevices() is
      // populated on mobile Chrome.
      let stream: MediaStream | null = null;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
        });
      } catch (camErr: unknown) {
        throw new Error(
          "Camera permission denied. Allow camera access and try again."
        );
      } finally {
        stream?.getTracks().forEach((t) => t.stop());
      }

      // Step 2: enumerate and pick the rear camera.
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((d) => d.kind === "videoinput");
      if (cameras.length === 0) {
        throw new Error(
          "No camera found on this device. Connect a webcam and try again."
        );
      }
      const rearCam =
        cameras.find((d) => /back|rear|environment/i.test(d.label || "")) ||
        cameras[cameras.length - 1];

      const scanner = new Html5Qrcode(elementId, { verbose: false });
      scannerRef.current = scanner;
      setOpen(true);

      // Allow React to commit the div to the DOM before html5-qrcode
      // looks it up by ID.
      await new Promise((resolve) => setTimeout(resolve, 0));

      try {
        await scanner.start(
          { deviceId: { exact: rearCam.deviceId } },
          { fps: 10, qrbox: { width: 250, height: 250 } },
          (payload: string) => {
            const value = payload.trim();
            if (!value) return;
            stopCamera();
            onScanRef.current(value);
          },
          () => {
            // Per-frame misses are normal; ignore.
          }
        );
      } catch (firstErr) {
        // Device-ID start can fail on some mobile browsers — retry with
        // facingMode, which is more universally supported.
        await scanner.start(
          { facingMode: "environment" },
          { fps: 10, qrbox: { width: 250, height: 250 } },
          (payload: string) => {
            const value = payload.trim();
            if (!value) return;
            stopCamera();
            onScanRef.current(value);
          },
          () => {
            // ignore
          }
        );
      }
    } catch (err: unknown) {
      setError(describeError(err));
      stopCamera();
    } finally {
      setStarting(false);
    }
  }

  function stopCamera() {
    const scanner = scannerRef.current;
    scannerRef.current = null;
    setOpen(false);
    if (!scanner) return;
    try {
      scanner.stop();
      scanner.clear();
    } catch {
      /* ignore */
    }
  }

  function describeError(err: unknown): string {
    const raw =
      typeof err === "string"
        ? err
        : err instanceof Error
          ? `${err.name}: ${err.message}`
          : JSON.stringify(err) || "unknown error";
    const lower = raw.toLowerCase();
    if (/permission|notallowed/i.test(lower)) {
      return "Camera permission denied. Allow camera access and try again.";
    }
    if (/notfound|no camera|not found|no device|no video/i.test(lower)) {
      return "No camera found on this device. Connect a webcam and try again.";
    }
    if (/constraint/i.test(lower)) {
      return "Camera constraint error. Try a different browser.";
    }
    return `Camera error: ${raw}`;
  }

  return (
    <div>
      {/* Always mounted: html5-qrcode does document.getElementById(id)
          at start(). If the div isn't in the DOM it throws and the
          scanner never opens. */}
      <div id={elementId} style={{ width: "100%", height: "260px", display: open ? "block" : "none" }} />
      {!open ? (
        <button
          type="button"
          onClick={startScan}
          disabled={disabled || starting}
          style={styles.scanButton}
        >
          {starting ? "Starting camera..." : "📷 Scan Household QR"}
        </button>
      ) : (
        <div>
          <p style={styles.hint}>
            Point the camera at the citizen's household QR code.
          </p>
          <button
            type="button"
            onClick={stopCamera}
            style={styles.cancelButton}
          >
            Cancel Scan
          </button>
        </div>
      )}
      {error && <p style={styles.error}>{error}</p>}
    </div>
  );
}

const styles = {
  scanButton: {
    marginTop: "10px",
    padding: "12px 18px",
    border: "none",
    borderRadius: "7px",
    background: "#15803d",
    color: "white",
    fontSize: "15px",
    fontWeight: "bold",
    cursor: "pointer",
    width: "100%",
  },
  cancelButton: {
    marginTop: "10px",
    padding: "8px 16px",
    border: "1px solid #d1d5db",
    borderRadius: "7px",
    background: "white",
    color: "#374151",
    fontSize: "14px",
    cursor: "pointer",
  },
  hint: {
    margin: "8px 0 0 0",
    fontSize: "13px",
    color: "#6b7280",
  },
  error: {
    margin: "8px 0 0 0",
    fontSize: "13px",
    color: "#b91c1c",
  },
};