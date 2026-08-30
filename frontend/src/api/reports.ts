import { apiRequest } from "./client";

export type Report = {
  id: string;
  description: string | null;
  address: string | null;
  latitude: number;
  longitude: number;
  status:
    | "PENDING"
    | "ACCEPTED"
    | "IN_PROGRESS"
    | "COMPLETED";
  image_url: string;
  created_at: string;
  updated_at: string;
};

export type ReportsResponse = {
  success: boolean;
  data: Report[];
};

export type ReportResponse = {
  success: boolean;
  data: Report;
};

export type ReportStatsResponse = {
  success: boolean;
  data: {
    total: number;
    pending: number;
    completed: number;
  };
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
) {
  const formData = new FormData();

  formData.append("image", image);
  formData.append("description", description);
  formData.append("address", address);
  formData.append("latitude", String(latitude));
  formData.append("longitude", String(longitude));

  return apiRequest<ReportResponse>(
    "/reports",
    {
      method: "POST",
      body: formData,
    }
  );
}

/*
 * Get all reports belonging to the logged-in citizen
 */
export async function getMyReports() {
  return apiRequest<ReportsResponse>(
    "/reports/my",
    {
      method: "GET",
    }
  );
}

/*
 * Get one specific report
 */
export async function getReport(
  reportId: string
) {
  return apiRequest<ReportResponse>(
    `/reports/${reportId}`,
    {
      method: "GET",
    }
  );
}

/*
 * Get report statistics for the logged-in citizen
 */
export async function getReportStats() {
  return apiRequest<ReportStatsResponse>(
    "/reports/stats",
    {
      method: "GET",
    }
  );
}
// ============================================================
// Worker API
// ============================================================

export async function getWorkerQueue() {
  return apiRequest<ReportsResponse>(
    "/reports/worker/queue",
    {
      method: "GET",
    }
  );
}

export async function getWorkerActiveReports() {
  return apiRequest<ReportsResponse>(
    "/reports/worker/active",
    {
      method: "GET",
    }
  );
}

export async function updateWorkerReportStatus(
  reportId: string,
  newStatus:
    | "ACCEPTED"
    | "IN_PROGRESS"
    | "COMPLETED"
) {
  return apiRequest<ReportResponse>(
    `/reports/worker/${reportId}/status?new_status=${newStatus}`,
    {
      method: "PATCH",
    }
  );
}
export type WorkerStatsResponse = {
  success: boolean;
  data: {
    pending: number;
    accepted: number;
    in_progress: number;
    completed_today: number;
  };
};

export async function getWorkerStats() {
  return apiRequest<WorkerStatsResponse>(
    "/reports/worker/stats",
    {
      method: "GET",
    }
  );
}
export type WorkerWard = {
  id: string;
  name: string;
  zone: string | null;
  description: string | null;
};

export type WorkerWardsResponse = {
  success: boolean;
  data: WorkerWard[];
};

export async function getWorkerWards() {
  return apiRequest<WorkerWardsResponse>(
    "/reports/worker/wards",
    {
      method: "GET",
    }
  );
}
// ============================================================
// Worker Schedule API
// ============================================================

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

export type WorkerSchedulesResponse = {
  success: boolean;
  data: WorkerSchedule[];
};

export async function getWorkerSchedules() {
  return apiRequest<WorkerSchedulesResponse>(
    "/reports/worker/schedules",
    {
      method: "GET",
    }
  );
}