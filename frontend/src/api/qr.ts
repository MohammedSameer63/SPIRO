import { apiRequest } from "./client";

export type HouseholdQr = {
  householdId: string;
  qrPayload: string;
  qrCode: string;
};

/*
 * Signed QR code for a household (citizens: own household only).
 *
 * Backend response:
 * { householdId, qrPayload, qrCode }
 * where qrCode is a data-URL PNG ready for an <img> tag.
 */
export async function getHouseholdQr(
  householdId: string
): Promise<HouseholdQr> {
  return apiRequest<HouseholdQr>(
    `/households/${householdId}/qr`,
    {
      method: "GET",
    }
  );
}

export type PickupValidationResult = {
  report_id: string;
  household_id: string;
  worker_id: string;
  method: string;
  validated_at: string | null;
};

/*
 * Validate a pickup by scanning the citizen's household QR code.
 * `qrPayload` is the raw string from the scan.
 */
export async function validatePickup(
  reportId: string,
  qrPayload: string
): Promise<{
  success: boolean;
  data: PickupValidationResult;
  message?: string;
}> {
  return apiRequest<{
    success: boolean;
    data: PickupValidationResult;
    message?: string;
  }>(`/worker/reports/${reportId}/validate-pickup`, {
    method: "POST",
    body: JSON.stringify({
      qr_payload: qrPayload,
    }),
  });
}
