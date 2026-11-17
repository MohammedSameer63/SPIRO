import base64
import hashlib
import hmac
import io
import json
import time
from uuid import UUID

from app.core.config import settings
from app.core.exceptions import InvalidQrPayloadError

# Payload envelope version. Bump if the signed format ever changes.
QR_PAYLOAD_VERSION = 1
QR_TOKEN_PREFIX = "sp1"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def build_household_qr_payload(household_id: UUID) -> str:
    """
    Build a signed QR payload for a household.

    Format:  sp1.<base64url-json>.<base64url-hmac-sha256>
    The signature covers the versioned body, so payloads cannot be
    forged or tampered with without the QR secret key. Kept compact
    (~140 chars) so the rendered QR stays at a low version and scans
    reliably on phone cameras.
    """
    body = {
        "v": QR_PAYLOAD_VERSION,
        "hid": str(household_id),
        "iat": int(time.time()),
    }
    body_b64 = _b64url_encode(
        json.dumps(body, separators=(",", ":")).encode("utf-8")
    )
    sig = hmac.new(
        settings.qr_secret_key.encode("utf-8"),
        f"{QR_TOKEN_PREFIX}.{body_b64}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{QR_TOKEN_PREFIX}.{body_b64}.{_b64url_encode(sig)}"


def verify_household_qr_payload(token: str) -> UUID:
    """
    Verify a scanned QR payload and return the household id it encodes.
    Raises InvalidQrPayloadError on any problem (format, signature, content).
    """
    try:
        prefix, body_b64, sig = token.strip().split(".")
    except ValueError:
        raise InvalidQrPayloadError("QR code format is invalid.")

    if prefix != QR_TOKEN_PREFIX:
        raise InvalidQrPayloadError("QR code is not a SPIRO household code.")

    expected_sig = hmac.new(
        settings.qr_secret_key.encode("utf-8"),
        f"{prefix}.{body_b64}".encode("ascii"),
        hashlib.sha256,
    ).digest()

    try:
        given_sig = _b64url_decode(sig)
    except Exception:
        raise InvalidQrPayloadError("QR code signature is invalid.")

    if not hmac.compare_digest(expected_sig, given_sig):
        raise InvalidQrPayloadError("QR code signature is invalid.")

    try:
        body = json.loads(_b64url_decode(body_b64))
    except Exception:
        raise InvalidQrPayloadError("QR code content is unreadable.")

    if body.get("v") != QR_PAYLOAD_VERSION:
        raise InvalidQrPayloadError("QR code version is not supported.")

    try:
        return UUID(str(body["hid"]))
    except Exception:
        raise InvalidQrPayloadError("QR code does not contain a valid household.")


def qr_image_data_url(payload: str) -> str:
    """
    Render the payload as a PNG QR code and return a data URL,
    ready for an <img> tag. Requires the `qrcode` package.
    """
    import qrcode

    # Generous module size + quiet zone: dense QRs with a thin border
    # fail to decode on real phone cameras (verified with OpenCV's
    # decoder — box_size=8/border=2 was unreliable, 10/4 is exact).
    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data(payload)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
