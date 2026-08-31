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
 *
 * Address, latitude and longitude are derived
 * from the citizen's household and are not
 * required.
 */
export async function createReport(
  image: File,
  description: string
): Promise<Report> {
  const formData = new FormData();

  formData.append("image", image);
  formData.append("description", description);

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

// ============================================================
// Worker-Ward API
// ============================================================

export type WorkerWard = {
  id: string;
  worker_id: string;
  ward_id: string;
  assigned_by: string;
  assigned_at: string;
  name: string;
  zone: string | null;
  description: string | null;
};

export async function getWorkerWards(): Promise<{
  success: boolean;
  data: WorkerWard[];
}> {
  return apiRequest<{ success: boolean; data: WorkerWard[] }>(
    "/worker/wards",
    {
      method: "GET",
    }
  );
}

export type WorkerSchedule = {
  id: string;
  ward_id: string;
  ward: string;
  category_id: string;
  category: string;
  day: string;
  startTime: string;
  endTime: string;
};

export async function getWorkerStats() {
  return apiRequest<{ success: boolean; data: { pending: number; accepted: number; in_progress: number; completed_today: number } }>(
    "/worker/dashboard", { method: "GET" }
  );
}

export async function getWorkerQueue() {
  return apiRequest<{ success: boolean; data: Report[] }>(
    "/worker/reports/queue", { method: "GET" }
  );
}

export async function getWorkerActiveReports() {
  return apiRequest<{ success: boolean; data: Report[] }>(
    "/worker/reports/active", { method: "GET" }
  );
}

export async function updateWorkerReportStatus(
  reportId: string,
  status: Report["status"]
) {
  return apiRequest<{ success: boolean; data: Report }>(
    `/worker/reports/${reportId}/status`,
    { method: "PATCH", body: JSON.stringify({ status }) }
  );
}

export async function getWorkerSchedules() {
  return apiRequest<{ success: boolean; data: WorkerSchedule[] }>(
    "/worker/schedules", { method: "GET" }
  );
}
