import { apiRequest } from "./client";

export type Report = {
  id: string;
  user_id: string;
  schedule_id: string | null;
  description: string | null;
  image_url: string;
  latitude: number | null;
  longitude: number | null;
  address: string | null;
  status:
    | "PENDING"
    | "ACCEPTED"
    | "IN_PROGRESS"
    | "COMPLETED";
  created_at: string;
  updated_at: string;
};

/*
 * Create a new waste report
 */
export async function createReport(
  image: File,
  description: string,
  address: string,
  latitude: number,
  longitude: number
): Promise<Report> {
  const formData = new FormData();

  formData.append("image", image);
  formData.append("description", description);
  formData.append("address", address);
  formData.append("latitude", String(latitude));
  formData.append("longitude", String(longitude));

  return apiRequest<Report>(
    "/reports",
    {
      method: "POST",
      body: formData,
    }
  );
}

/*
 * Get all reports belonging to the logged-in citizen
 *
 * Backend response:
 * [
 *   WasteReportResponse,
 *   WasteReportResponse,
 *   ...
 * ]
 */
export async function getMyReports(): Promise<Report[]> {
  return apiRequest<Report[]>(
    "/reports/my",
    {
      method: "GET",
    }
  );
}

/*
 * Get one specific report
 *
 * Backend response:
 * WasteReportResponse
 */
export async function getReport(
  reportId: string
): Promise<Report> {
  return apiRequest<Report>(
    `/reports/${reportId}`,
    {
      method: "GET",
    }
  );
}