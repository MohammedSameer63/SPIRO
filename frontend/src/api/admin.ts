import { apiRequest } from "./client";

export type AdminDashboardResponse = {
  success: boolean;

  data: {
    stats: {
      total_citizens: number;
      total_workers: number;
      total_reports: number;
      pending_reports: number;
      completed_reports: number;
    };

    workers: {
      id: string;
      name: string;
      email: string;
      active_reports: number;
      completed_today: number;
    }[];

    wards: {
      ward: string;
      reports: number;
      pending: number;
    }[];
  };
};

export type AdminWorker = {
  id: string;
  name: string;
  email: string;
  status: string;

  wards: {
    id: string;
    name: string;
  }[];
};

export type AdminWard = {
  id: string;
  name: string;
  zone: string | null;
  description: string | null;
};

export type AdminWorkersResponse = {
  success: boolean;
  data: AdminWorker[];
};

export type AdminWardsResponse = {
  success: boolean;
  data: AdminWard[];
};

export type AssignWorkerResponse = {
  success: boolean;
  message: string;

  data: {
    worker_id: string;
    ward_id: string;
    ward_name: string;
  };
};

export async function getAdminDashboard() {
  return apiRequest<AdminDashboardResponse>(
    "/admin/dashboard",
    {
      method: "GET",
    }
  );
}

export async function getAdminWorkers() {
  return apiRequest<AdminWorkersResponse>(
    "/admin/workers",
    {
      method: "GET",
    }
  );
}

export async function getAdminWards() {
  return apiRequest<AdminWardsResponse>(
    "/admin/wards",
    {
      method: "GET",
    }
  );
}

export async function assignWorkerToWard(
  workerId: string,
  wardId: string
) {
  return apiRequest<AssignWorkerResponse>(
    `/admin/workers/${workerId}/wards/${wardId}`,
    {
      method: "POST",
    }
  );
}

export async function removeWorkerFromWard(
  workerId: string,
  wardId: string
) {
  return apiRequest<{
    success: boolean;
    message: string;
    data: { worker_id: string; ward_id: string };
  }>(
    `/admin/workers/${workerId}/wards/${wardId}`,
    { method: "DELETE" }
  );
}

// NOTE: removed stale helpers that called backend routes which do not
// exist (GET /reports/admin/pending, POST /reports/admin/.../assign/...).
// Admins assign via POST /assignments; pending counts come from
// GET /admin/dashboard. Do not re-add without the backend endpoint.

// ============================================================
// Admin Schedule API
// ============================================================

export type AdminSchedule = {
  id: string;
  ward_id: string;
  ward: string;
  category_id: string;
  category: string;
  day: string;
  startTime: string;
  endTime: string;
};

export type AdminSchedulesResponse = {
  success: boolean;
  data: AdminSchedule[];
};

export async function getAdminSchedules() {
  return apiRequest<AdminSchedulesResponse>(
    "/admin/schedules",
    {
      method: "GET",
    }
  );
}

export async function createAdminSchedule(
  wardId: string,
  categoryId: string,
  day: string,
  startTime: string,
  endTime: string
) {
  return apiRequest<{
    success: boolean;
    message: string;
    data: AdminSchedule;
  }>(
    "/admin/schedules",
    {
      method: "POST",
      body: JSON.stringify({
        ward_id: wardId,
        waste_category_id: categoryId,
        day_of_week: day,
        start_time: startTime,
        end_time: endTime,
      }),
    }
  );
}

export async function deleteAdminSchedule(
  scheduleId: string
) {
  return apiRequest<{
    success: boolean;
    message: string;
  }>(
    `/admin/schedules/${scheduleId}`,
    {
      method: "DELETE",
    }
  );
}
export type AdminWasteCategory = {
  id: string;
  name: string;
  description: string | null;
};

export type AdminWasteCategoriesResponse = {
  success: boolean;
  data: AdminWasteCategory[];
};

export async function getAdminWasteCategories() {
  return apiRequest<AdminWasteCategoriesResponse>(
    "/admin/waste-categories",
    {
      method: "GET",
    }
  );
}

export type AdminReport = {
  id: string;
  user_id: string;
  schedule_id: string | null;
  description: string | null;
  image_url: string;
  latitude: number | null;
  longitude: number | null;
  address: string | null;
  status: "PENDING" | "ACCEPTED" | "IN_PROGRESS" | "COMPLETED";
  created_at: string;
  updated_at: string;
  prediction: {
    id: string;
    category_id: string;
    category_name: string | null;
    confidence: number | string | null;
    model_used: string | null;
    prediction_time: string;
  } | null;
  pickup_validated?: boolean;
  coins_awarded?: number;
};

/*
 * All reports, newest first (admin audit view).
 */
export async function getAdminReports(status?: string, limit = 100) {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  params.set("limit", String(limit));
  return apiRequest<{ success: boolean; data: AdminReport[] }>(
    `/admin/reports?${params.toString()}`,
    { method: "GET" }
  );
}

export type MlStatus = {
  enabled: boolean;
  initialized: boolean;
  mock_mode: boolean;
  model_file: string;
  architecture: string;
  input_size: number;
  num_classes: number;
  confidence_threshold: number;
  pipeline_stages: string[];
};

/*
 * Live ML service state for the model registry view.
 */
export async function getMlStatus() {
  return apiRequest<{ success: boolean; data: MlStatus }>(
    "/ml/status",
    { method: "GET" }
  );
}
