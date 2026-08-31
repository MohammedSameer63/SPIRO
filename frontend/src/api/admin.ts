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

export async function getAdminPendingReports() {
  return apiRequest<{
    success: boolean;
    data: {
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
    }[];
 }>("/reports/admin/pending", {
    method: "GET",
  });
}

export async function assignReportToWorker(
  reportId: string,
  workerId: string
) {
  return apiRequest<{
    success: boolean;
    message: string;
    data: {
      report_id: string;
      worker_id: string;
      worker_name: string;
    };
  }>(
    `/reports/admin/${reportId}/assign/${workerId}`,
    {
      method: "POST",
    }
  );
}
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
