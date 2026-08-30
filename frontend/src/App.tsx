import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";

import Login from "./pages/auth/Login";
import Register from "./pages/auth/Register";

import CitizenDashboard from "./pages/citizen/Dashboard";
import SubmitReport from "./pages/citizen/SubmitReport";
import Reports from "./pages/citizen/Reports";
import ReportDetails from "./pages/citizen/ReportDetails";

import CitizenLayout from "./layouts/CitizenLayout";
import WorkerLayout from "./layouts/WorkerLayout";
import AdminLayout from "./layouts/AdminLayout";

import WorkerDashboard from "./pages/worker/Dashboard";
import WorkerQueue from "./pages/worker/Queue";
import ActiveReports from "./pages/worker/ActiveReports";
import WorkerSchedules from "./pages/worker/Schedules";

import AdminDashboard from "./pages/admin/Dashboard";
import AdminWorkers from "./pages/admin/Workers";
import AdminSchedules from "./pages/admin/Schedules";


function App() {
  return (
    <BrowserRouter>
      <Routes>

        {/* Authentication */}
        <Route
          path="/login"
          element={<Login />}
        />

        <Route
          path="/register"
          element={<Register />}
        />

        {/* Citizen */}
        <Route
          path="/citizen"
          element={<CitizenLayout />}
        >
          <Route
            path="dashboard"
            element={<CitizenDashboard />}
          />
          <Route
            path="report"
            element={<SubmitReport />}
          />
          <Route
            path="reports"
            element={<Reports />}
          />
          <Route
            path="reports/:id"
            element={<ReportDetails />}
          />
        </Route>
       {/* Worker */}
<Route
  path="/worker"
  element={<WorkerLayout />}
>
  <Route
    path="dashboard"
    element={<WorkerDashboard />}
  />

  <Route
    path="queue"
    element={<WorkerQueue />}
  />

  <Route
    path="active"
    element={<ActiveReports />}
  />

  <Route
    path="schedules"
    element={<WorkerSchedules />}
  />
</Route>
        {/* Admin */}

         <Route
            path="/admin"
            element={<AdminLayout />}
          >
          <Route
              path="dashboard"
             element={<AdminDashboard />}
          />

          <Route
             path="workers"
             element={<AdminWorkers />}
          />

          <Route
              path="schedules"
             element={<AdminSchedules />}
          />
         </Route>

        {/* Default */}
        <Route
          path="/"
          element={<Navigate to="/login" replace />}
        />

        {/* Unknown URL */}
        <Route
          path="*"
          element={<Navigate to="/login" replace />}
        />

      </Routes>
    </BrowserRouter>
  );
}

export default App;