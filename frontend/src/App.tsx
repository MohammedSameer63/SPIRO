import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";

import Login from "./pages/auth/Login";
import Register from "./pages/auth/Register";
import Landing from "./pages/Landing";

import CitizenDashboard from "./pages/citizen/Dashboard";
import SubmitReport from "./pages/citizen/SubmitReport";
import Reports from "./pages/citizen/Reports";
import ReportDetails from "./pages/citizen/ReportDetails";
import Wallet from "./pages/citizen/Wallet";
import MyQR from "./pages/citizen/MyQR";
import Alerts from "./pages/citizen/Alerts";

import CitizenLayout from "./layouts/CitizenLayout";
import WorkerLayout from "./layouts/WorkerLayout";
import AdminLayout from "./layouts/AdminLayout";
import { RequireRole } from "./components/RequireRole";

import WorkerDashboard from "./pages/worker/Dashboard";
import WorkerQueue from "./pages/worker/Queue";
import ActiveReports from "./pages/worker/ActiveReports";
import Verify from "./pages/worker/Verify";
import RoutePage from "./pages/worker/Route";
import WorkerSchedules from "./pages/worker/Schedules";

import AdminDashboard from "./pages/admin/Dashboard";
import AdminAudit from "./pages/admin/Audit";
import AdminWorkers from "./pages/admin/Workers";
import AdminWards from "./pages/admin/Wards";
import AdminSchedules from "./pages/admin/Schedules";
import AdminML from "./pages/admin/ML";


function App() {
  return (
    <BrowserRouter>
      <Routes>

        {/* Public landing */}
        <Route
          path="/"
          element={<Landing />}
        />

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
        <Route element={<RequireRole roles={["CITIZEN"]} />}>
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
          <Route
            path="wallet"
            element={<Wallet />}
          />
          <Route
            path="qr"
            element={<MyQR />}
          />
          <Route
            path="alerts"
            element={<Alerts />}
          />
        </Route>
        </Route>
       {/* Worker */}
<Route element={<RequireRole roles={["WORKER"]} />}>
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
    path="verify/:id"
    element={<Verify />}
  />

  <Route
    path="route"
    element={<RoutePage />}
  />

  <Route
    path="schedules"
    element={<WorkerSchedules />}
  />
</Route>
</Route>
        {/* Admin */}

         <Route element={<RequireRole roles={["ADMIN"]} />}>
          <Route
            path="/admin"
            element={<AdminLayout />}
          >
          <Route
              path="dashboard"
             element={<AdminDashboard />}
          />

          <Route
              path="audit"
             element={<AdminAudit />}
          />

          <Route
             path="workers"
            element={<AdminWorkers />}
          />

          <Route
             path="wards"
            element={<AdminWards />}
          />

          <Route
              path="schedules"
             element={<AdminSchedules />}
          />

          <Route
              path="ml"
             element={<AdminML />}
          />
          </Route>
         </Route>

        {/* Unknown URL */}
        <Route
          path="*"
          element={<Navigate to="/" replace />}
        />

      </Routes>
    </BrowserRouter>
  );
}

export default App;
