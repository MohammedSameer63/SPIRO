# SPIRO REST API Documentation

**Version:** 1.0 (MVP)

**Status:** Working Contract (MVP)

This document is the working API contract. Implemented behavior takes precedence over older draft examples; implementation changes are reflected here as development progresses.

---

# Overview

SPIRO is an AI-powered smart waste management system that enables citizens to report waste, workers to manage collections, and administrators to monitor city-wide waste management operations.

This document defines the REST API contract between the frontend (React/React Native), backend (FastAPI), and AI services.

---

# API Information

| Property | Value |
|----------|-------|
| API Style | REST |
| Base URL | `/api/v1` |
| Backend Framework | FastAPI |
| Authentication | JWT Bearer Token |
| Request Format | `application/json` |
| File Upload | `multipart/form-data` |
| Response Format | JSON |

---

# Standard Response Format

## Success Response

```json
{
    "success": true,
    "message": "Operation completed successfully.",
    "data": {}
}
```

---

## Error Response

```json
{
    "success": false,
    "message": "Invalid request."
}
```

---

# Authentication & Authorization

The API supports Role-Based Access Control (RBAC).

| Role | Description |
|------|-------------|
| Citizen | Registered household member |
| Worker | Waste collection worker |
| Admin | Municipality administrator |

---

# HTTP Status Codes

| Status Code | Meaning |
|-------------|----------|
| 200 | OK |
| 201 | Resource Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Resource Not Found |
| 409 | Conflict |
| 422 | Validation Error |
| 500 | Internal Server Error |

---

# API Modules

This document is organized feature-by-feature.

1. Authentication
2. Ward & Household
3. Waste Reporting
4. Workforce Management
5. Categories
6. Collection Schedule
7. Dashboard
8. Predictions

---

# Frontend Integration Endpoints

These additive endpoints support the current web client. They do not replace
the existing module endpoints documented elsewhere in this contract.

All endpoints below require a JWT bearer token and return the standard
`{ "success": true, "data": ... }` envelope.

## Workflow rules

- An administrator assigns workers to wards; a worker may have at most five
  ward assignments.
- A collection schedule belongs to a ward and waste category. It is not an
  assignment to an individual worker.
- Workers see pending reports from their assigned wards and accept those
  reports themselves.
- Once accepted, only the accepting worker may advance a report from
  `ACCEPTED` to `IN_PROGRESS` and then `COMPLETED`.

## Admin

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/admin/dashboard` | City totals plus ward and worker summaries. |
| GET | `/admin/workers` | Workers and their assigned wards. |
| GET | `/admin/wards` | Wards for worker assignment and schedules. |
| POST | `/admin/workers/{worker_id}/wards/{ward_id}` | Assign a worker to a ward. |
| DELETE | `/admin/workers/{worker_id}/wards/{ward_id}` | Remove a worker from a ward. |
| GET | `/admin/schedules` | List all ward collection schedules. |
| POST | `/admin/schedules` | Create a ward/category collection schedule. |
| DELETE | `/admin/schedules/{schedule_id}` | Delete a collection schedule. |
| GET | `/admin/waste-categories` | List waste categories for schedules. |

`POST /admin/schedules` accepts:

```json
{
  "ward_id": "UUID",
  "waste_category_id": "UUID",
  "day_of_week": "MONDAY",
  "start_time": "09:00:00",
  "end_time": "12:00:00"
}
```

## Worker

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/worker/dashboard` | Queue and assignment counts for the authenticated worker. |
| GET | `/worker/wards` | The worker's assigned wards. |
| GET | `/worker/reports/queue` | Pending reports whose reporting household ward is explicitly assigned to the authenticated worker. |
| GET | `/worker/reports/active` | Reports accepted by the worker that remain active. |
| PATCH | `/worker/reports/{report_id}/status` | Accept or advance a report. |
| GET | `/worker/schedules` | Schedules for the worker's assigned wards. |

`PATCH /worker/reports/{report_id}/status` accepts exactly one of these valid
transitions:

```json
{ "status": "ACCEPTED" }
```

for an unassigned `PENDING` report in an assigned ward, then:

```json
{ "status": "IN_PROGRESS" }
```

and finally:

```json
{ "status": "COMPLETED" }
```

No database migration is required: report ownership is derived from
`users.household_id -> households.ward_id`, worker eligibility from
`worker_wards`, and acceptance from `assignments`.

---

# Feature 1 — Authentication

Authentication handles citizen registration and login. JWT access tokens are used for protected APIs.

## Implemented Endpoints

- `POST /auth/register` — Implemented and tested
- `POST /auth/login` — Implemented and tested
- `POST /auth/logout` — Planned
- `GET /auth/me` — Planned

---

## POST `/auth/register`

### Purpose

Register a new citizen account.

If a household matching the supplied location already exists, the new citizen is linked to that household. A new household is created only when no matching household exists.

### Authentication Required

**No**

### Request Body

```json
{
  "name": "Sameer",
  "email": "sameer@example.com",
  "password": "Password@123",
  "phone": "9876543210",
  "ward_id": "UUID",
  "house_number": "12A",
  "street_name": "MG Road",
  "address": "12A MG Road, Bengaluru"
}
```

> The external field names currently follow the backend Pydantic schemas (`snake_case`).

### Validation Rules

#### User

| Field | Rules |
|-------|-------|
| name | Required |
| email | Required, valid email format, unique |
| password | Required; password hashing is applied before storage |
| phone | Optional |

#### Household

| Field | Rules |
|-------|-------|
| ward_id | Required, must exist |
| house_number | Required |
| street_name | Optional |
| address | Required |

### Business Rules

- Selected ward must exist.
- Email must be unique.
- New users are assigned the `CITIZEN` role.
- New users start with `ACTIVE` status.
- A matching household is reused when registering another citizen at the same household.
- Multiple citizens may belong to the same household.
- The citizen is linked to the household through `household_id`.
- Passwords are stored only as hashes.
- Household QR generation is **not part of the implemented registration flow yet**.

### Success Response

**201 Created**

```json
{
  "access_token": "JWT_TOKEN",
  "token_type": "bearer",
  "user": {
    "id": "UUID",
    "name": "Sameer",
    "email": "sameer@example.com",
    "phone": "9876543210",
    "role": "CITIZEN",
    "status": "ACTIVE"
  }
}
```

### Error Responses

| Status | Description |
|--------|-------------|
| 404 | Ward not found |
| 409 | Email already registered |
| 422 | Request validation failed |
| 500 | Internal server error |

> A duplicate household is **not** an error during registration. An existing matching household is reused.

---

## POST `/auth/login`

### Purpose

Authenticate an existing user and return a JWT access token.

### Authentication Required

**No**

### Request Body

```json
{
  "email": "john@example.com",
  "password": "Password@123"
}
```

### Validation Rules

| Field | Rules |
|------|-------|
| email | Required, valid email format |
| password | Required |

### Success Response

**200 OK**

```json
{
  "access_token": "JWT_TOKEN",
  "token_type": "bearer",
  "user": {
    "id": "UUID",
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "9876543210",
    "role": "CITIZEN",
    "status": "ACTIVE"
  }
}
```

### Error Responses

| Status | Description |
|---------|-------------|
| 401 | Invalid email or password |
| 403 | Account is not active |

### Account Status Rules

| User Status | Login |
|-------------|-------|
| `ACTIVE` | Allowed |
| `INACTIVE` | Rejected with `403` |
| `SUSPENDED` | Rejected with `403` |

### Notes

- JWT expires after **24 hours**.
- JWT type is `bearer`.
- Invalid email and wrong password intentionally return the same `401` response.
- `password_hash` is never included in the response.

---

## POST `/auth/logout`

### Status

**Planned**

JWT logout will initially be handled client-side by removing the access token. Server-side invalidation can be introduced later if refresh tokens or token blacklisting are added.

---

## GET `/auth/me`

### Status

**Planned**

This endpoint will use the JWT access token to identify and return the currently authenticated user.

---

## Authentication Flow

```text
Register
      │
      ▼
Login
      │
      ▼
Receive JWT
      │
      ▼
Store Token
      │
      ▼
Authorization: Bearer <JWT_TOKEN>
      │
      ▼
Access Protected APIs
```

---

## Authentication Rules

### Password

- Password is hashed before storage.
- Bcrypt is currently used by the backend.
- Password hashes are never returned through the API.

### Email

- Must be unique.
- Must be a valid email address.

### Phone

- Optional.

### JWT

- Expiry: 24 hours.
- Type: Bearer token.

### Authorization Header

```http
Authorization: Bearer <JWT_TOKEN>
```

# ============================================
# Feature 2 - Ward & Household
# ============================================

## Overview

This module manages municipal wards and registered households.

A household belongs to one ward.

Citizens are registered under a household.

Workers and Admins are not associated with a household.

---

# Ward Hierarchy

Ward
└── Household
      └── Citizen
            └── Waste Reports

---

# Permissions

| Role | Access |
|------|--------|
| Citizen | Read only |
| Worker | Read only |
| Admin | Full access |

---

# ============================================
# GET /wards
# ============================================

## Purpose

Returns all municipality wards.

## Authentication

Citizen

Worker

Admin

## Request

No request body.

## Success

200 OK

```json
{
  "success": true,
  "data": [
    {
      "id": "UUID",
      "name": "Ward 12",
      "zone": "North",
      "description": "North Municipal Zone"
    }
  ]
}
```

## Errors

| Status | Meaning |
|---------|---------|
|401|Unauthorized|

---

# ============================================
# GET /wards/{id}
# ============================================

## Purpose

Returns details of a single ward.

## Authentication

Citizen

Worker

Admin

## Path Parameter

```
id : UUID
```

## Success

```json
{
  "success": true,
  "data": {
    "id":"UUID",
    "name":"Ward 12",
    "zone":"North",
    "description":"North Municipal Zone"
  }
}
```

## Errors

|Status|Meaning|
|------|-------|
|404|Ward Not Found|

---

# ============================================
# GET /households
# ============================================

## Purpose

Returns households.

Citizen:
Only own household.

Admin:
All households.

Worker:
Read-only access if required.

## Authentication

Citizen

Worker

Admin

## Query Parameters

| Parameter | Description |
|-----------|-------------|
|wardId|Filter households by ward|

Example

```
GET /households?wardId=<UUID>
```

---

# ============================================
# POST /households
# ============================================

## Purpose

Create a new household.

Only Admin can create households.

## Authentication

Admin

## Request

```json
{
    "wardId":"UUID",
    "houseNumber":"12A",
    "streetName":"MG Road",
    "address":"12A MG Road Bengaluru"
}
```

## Success

201 Created

```json
{
    "success":true,
    "message":"Household created successfully.",
    "data":{
        "householdId":"UUID"
    }
}
```

## Errors

|Status|Meaning|
|------|-------|
|400|Validation Failed|
|404|Ward Not Found|

---

# ============================================
# GET /households/{id}
# ============================================

## Purpose

Returns details of a household.

Citizen

Can access only their household.

Admin

Can access any household.

## Authentication

Citizen

Admin

## Success

```json
{
    "success":true,
    "data":{
        "id":"UUID",
        "wardId":"UUID",
        "houseNumber":"12A",
        "streetName":"MG Road",
        "address":"12A MG Road Bengaluru"
    }
}
```

---

# ============================================
# PATCH /households/{id}
# ============================================

## Purpose

Update household information.

## Authentication

Admin

## Request

```json
{
    "houseNumber":"15B",
    "streetName":"Church Street",
    "address":"15B Church Street Bengaluru"
}
```

## Success

200 OK

```json
{
    "success":true,
    "message":"Household updated successfully."
}
```

## Errors

|Status|Meaning|
|------|-------|
|400|Validation Failed|
|404|Household Not Found|

---

# ============================================
# GET /households/{id}/qr
# ============================================

## Purpose

Returns household QR information.

Version 1

Read-only endpoint.

QR generation happens automatically during household creation.

## Authentication

Citizen

Worker

Admin

## Success

```json
{
    "success":true,
    "data":{
        "householdId":"UUID",
        "qrCode":"BASE64_OR_URL"
    }
}
```

## Errors

|Status|Meaning|
|------|-------|
|404|Household Not Found|

---

# Validation Rules

Ward

- Name must be unique.

Household

- House Number is mandatory.
- Every household belongs to exactly one ward.
- Household address is mandatory.

---

# Authorization Summary

| Endpoint | Citizen | Worker | Admin |
|-----------|---------|--------|-------|
|GET /wards|✅|✅|✅|
|GET /wards/{id}|✅|✅|✅|
|GET /households|Own|Read|All|
|POST /households|❌|❌|✅|
|GET /households/{id}|Own|Read|All|
|PATCH /households/{id}|❌|❌|✅|
|GET /households/{id}/qr|✅|✅|✅|




# ============================================
# Feature 3 - Waste Reporting
# ============================================

## Overview

The Waste Reporting module allows citizens to report waste by uploading an image and providing the waste location.

After a report is submitted:

1. Backend validates the request.
2. Image is stored.
3. Report is created.
4. ML Service classifies the waste.
5. Prediction is stored.
6. Report becomes available to Admin.
7. Admin assigns a Worker.

---

# Report Lifecycle

PENDING
    ↓
ACCEPTED
    ↓
IN_PROGRESS
    ↓
COMPLETED

---

# Permissions

| Role    | Access                                         |
| ------- | ---------------------------------------------- |
| Citizen | Create & View Own Reports                      |
| Worker  | View Reports in Assigned Wards & Update Status |
| Admin   | View All Reports & Monitor Operations          |

---

# ============================================
# POST /reports
# ============================================

## Purpose

Create a new waste report.

This endpoint also initiates the ML classification workflow.

---

## Authentication

Citizen

---

## Headers

Authorization: Bearer <JWT>

Content-Type: multipart/form-data

---

## Request

| Field | Type | Required |
|-------|------|----------|
| image | File | Yes |
| description | String | No |

Latitude, longitude and address are no longer required. When omitted, the report's address is automatically derived from the citizen's household address (house number + street name + address) registered at signup.

---

## Validation Rules

Image

- JPG
- JPEG
- PNG

Maximum size

10 MB

Location

Optional

Latitude

-90 → +90

Longitude

-180 → +180

Address

Derived from the citizen's household when not supplied

---

## Backend Workflow

Citizen

↓

Validate JWT

↓

Validate Image

↓

Run Edge AI (MobileNet)

↓

Store Image

↓

Create Report

↓

Store Prediction

↓

Return Response

---

## Success Response

201 Created

```json
{
    "success": true,
    "message": "Waste report submitted successfully.",
    "data": {
        "reportId": "UUID",
        "status": "PENDING",
        "prediction": {
            "category": "Plastic",
            "confidence": 94.20
        }
    }
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
|400|Validation Failed|
|401|Unauthorized|
|413|Image Too Large|
|415|Unsupported Image Format|
|500|Internal Server Error|

---

# ============================================
# GET /reports
# ============================================

## Purpose

Returns reports based on user role.

Citizen

Returns only own reports.


Worker

Returns all reports from the worker's assigned ward(s).


Admin

Returns all reports.

---

## Authentication

Citizen

Worker

Admin

---

## Query Parameters

| Parameter | Description |
|----------|-------------|
|status|Filter by report status|
|page|Pagination|
|limit|Pagination|

Example

GET /reports?status=PENDING&page=1&limit=10

---

## Success Response

```json
{
    "success": true,
    "data": [
        {
            "reportId":"UUID",
            "status":"PENDING",
            "category":"Plastic",
            "createdAt":"2026-07-27T10:30:00Z"
        }
    ]
}
```

---

# ============================================
# GET /reports/{id}
# ============================================

## Purpose

Returns complete details of a report.

---

## Authentication

Citizen

Worker

Admin

---

## Authorization Rules

Citizen

Can access only their reports.


Worker

Can access only assigned reports.


Admin

Can access all reports.

---

## Success Response

```json
{
    "success": true,
    "data": {
        "reportId":"UUID",

        "description":"Garbage near park",

        "imageUrl":"https://...",

        "location":{
            "latitude":12.9716,
            "longitude":77.5946,
            "address":"MG Road"
        },

        "status":"ASSIGNED",

        "prediction":{
            "category":"Plastic",
            "confidence":94.20
        },

        "createdAt":"2026-07-27T10:30:00Z"
    }
}
```

---

# ============================================
# PATCH /reports/{id}/status
# ============================================

## Purpose

Update report status.

---

## Authentication

Worker

---

## Allowed Status Flow

PENDING

↓

ACCEPTED

↓

IN_PROGRESS

↓

COMPLETED

---

## Request

```json
{
    "status":"IN_PROGRESS"
}
```

---

## Success Response

```json
{
    "success": true,
    "message":"Report status updated successfully."
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
|400|Invalid Status Transition|
|401|Unauthorized|
|403|Forbidden|
|404|Report Not Found|

---

# ============================================
# POST /reports/{id}/accept
# ============================================

## Purpose

Allows a worker assigned to the report's ward to accept responsibility for a waste report.

Once accepted:

- An assignment record is created.
- The report is assigned to the worker.
- Report status changes to **ACCEPTED**.

---

## Authentication

Worker

---

## Authorization Rules

- Worker must belong to the same ward as the report.
- Report must be in **PENDING** status.
- A report can only be accepted once.

---

## Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| id | UUID | Report ID |

---

## Request Body

No request body required.

---

## Backend Workflow

Worker

↓

Validate JWT

↓

Verify Worker Role

↓

Verify Worker belongs to Report Ward

↓

Verify Report Status = PENDING

↓

Create Assignment

↓

Update Report Status = ACCEPTED

↓

Return Success

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "message": "Report accepted successfully.",
  "data": {
    "reportId": "UUID",
    "workerId": "UUID",
    "status": "ACCEPTED"
  }
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
|401|Unauthorized|
|403|Worker not assigned to this ward|
|404|Report not found|
|409|Report already accepted|
|409|Invalid report status|

---

## Business Rules

- Only workers can accept reports.
- Workers may only accept reports from their assigned ward(s).
- Once accepted, the report becomes unavailable for other workers.
- Acceptance automatically creates an assignment record.

---

# Overall Business Rules

- Citizens can create unlimited reports.
- A report belongs to exactly one citizen.
- Every report has one prediction.

- Workers are assigned to one or more wards.
A worker can work only on reports belonging to their assigned ward(s).
Once a worker accepts a report, it becomes assigned to that worker.

- Reports cannot be deleted.
- Completed reports become read-only.

- Edge AI (MobileNet) performs classification during report submission.
Prediction is stored together with the report.

---

# Authorization Summary

| Endpoint                     | Citizen | Worker       | Admin        |
| ---------------------------- | ------- | ------------ | ------------ |
| POST /reports                | ✅      | ❌           | ❌           |
| GET /reports                 | Own     | Ward Reports | All          |
| GET /reports/{id}            | Own     | Ward Reports | All          |
| PATCH /reports/{id}/status   | ❌      | Ward Reports | Monitor Only |
| GET /reports/{id}/prediction | Own     | Ward Reports | All          |




# ============================================
# Feature 4 - Workforce Management
# ============================================

## Overview

The Workforce Management module manages municipal workers and their assigned wards.

Workers can:

- View reports from their assigned ward(s).
- Accept responsibility for pending reports.
- Update report progress.
- View their active and completed work.

Administrators can:

- Assign workers to wards.
- View workforce distribution.
- Monitor worker workloads.

---

# Workforce Flow

Admin

↓

Assign Worker to Ward

↓

Worker Logs In

↓

Views Pending Reports

↓

Accepts Report

↓

Updates Progress

↓

Completes Collection

---

# Permissions

| Role | Access |
|------|--------|
| Citizen | No Access |
| Worker | View Assigned Wards, Accept Reports, Update Progress |
| Admin | Manage Worker-Ward Assignments |

---

# ============================================
# GET /workers/me/wards
# ============================================

## Purpose

Returns all wards assigned to the logged-in worker.

---

## Authentication

Worker

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "data": [
    {
      "wardId": "UUID",
      "name": "Ward 12",
      "zone": "North Zone"
    }
  ]
}
```

---

# ============================================
# GET /workers/me/queue
# ============================================

## Purpose

Returns all reports available in the worker's assigned ward(s).

Pending reports are visible until another worker accepts them.

---

## Authentication

Worker

---

## Query Parameters

| Parameter | Description |
|-----------|-------------|
| status | Filter by report status |
| page | Page number |
| limit | Page size |

Example

GET /workers/me/reports?status=PENDING&page=1&limit=20

---

## Success Response

```json
{
  "success": true,
  "data": [
    {
      "reportId": "UUID",
      "status": "PENDING",
      "address": "12A MG Road",
      "prediction": "Plastic",
      "createdAt": "2026-07-27T10:30:00Z"
    }
  ]
}
```

---

# ============================================
# GET /workers/me/active
# ============================================

## Purpose

Returns reports currently accepted by the logged-in worker.

---

## Authentication

Worker

---

## Success Response

```json
{
  "success": true,
  "data": [
    {
      "reportId": "UUID",
      "status": "IN_PROGRESS",
      "acceptedAt": "2026-07-27T11:00:00Z"
    }
  ]
}
```

---

# ============================================
# GET /workers/me/history
# ============================================

## Purpose

Returns completed reports handled by the logged-in worker.

---

## Authentication

Worker

---

## Success Response

```json
{
  "success": true,
  "data": [
    {
      "reportId": "UUID",
      "completedAt": "2026-07-27T12:15:00Z",
      "address": "MG Road"
    }
  ]
}
```

---

# ============================================
# POST /admin/workers/{workerId}/wards
# ============================================

## Purpose

Assign one or more wards to a worker.

---

## Authentication

Admin

---

## Request Body

```json
{
  "wardIds": [
    "UUID",
    "UUID"
  ]
}
```

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "message": "Worker assigned to wards successfully."
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
|400|Validation Failed|
|404|Worker Not Found|
|404|Ward Not Found|

---

# ============================================
# GET /admin/workers
# ============================================

## Purpose

Returns all registered workers and their assigned wards.

---

## Authentication

Admin

---

## Success Response

```json
{
  "success": true,
  "data": [
    {
      "workerId": "UUID",
      "name": "John",
      "wards": [
        "Ward 12",
        "Ward 15"
      ]
    }
  ]
}
```

---

# Business Rules

### Workforce Rules

- Every worker must be assigned to at least one ward.
- A worker can be assigned to a maximum of **5 wards**.
- Multiple workers may be assigned to the same ward.
- A worker cannot be assigned to the same ward more than once.
- Only administrators can assign workers to wards.
- Workers may only view reports from their assigned wards.
- A report can only be accepted once.
- Accepting a report automatically creates an assignment record.
- Completed reports become read-only.

---

# Authorization Summary

| Endpoint | Citizen | Worker | Admin |
|-----------|---------|--------|-------|
|GET /workers/me/wards|❌|✅|❌|
|GET /workers/me/reports|❌|✅|❌|
|GET /workers/me/active|❌|✅|❌|
|GET /workers/me/history|❌|✅|❌|
|POST /admin/workers/{workerId}/wards|❌|❌|✅|
|GET /admin/workers|❌|❌|✅|




# ============================================
# Feature 5 - Waste Categories
# ============================================

## Overview

The Waste Categories module provides the list of supported waste categories used throughout SPIRO.

These categories are predefined and correspond to the labels supported by the AI classification model.

Categories are read-only in Version 1.

---

# Permissions

| Role | Access |
|------|--------|
| Citizen | View |
| Worker | View |
| Admin | View |

---

# ============================================
# GET /categories
# ============================================

## Purpose

Returns all supported waste categories.

These categories are used for:

- AI prediction
- Waste reports
- Collection schedules
- Dashboard statistics

---

## Authentication

Citizen

Worker

Admin

---

## Request

No request body.

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "data": [
    {
      "id": "UUID",
      "name": "Plastic",
      "recyclable": true,
      "description": "Plastic waste"
    },
    {
      "id": "UUID",
      "name": "Paper",
      "recyclable": true,
      "description": "Paper waste"
    },
    {
      "id": "UUID",
      "name": "Organic",
      "recyclable": false,
      "description": "Organic waste"
    }
  ]
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
|401|Unauthorized|
|500|Internal Server Error|

---

# Business Rules

- Categories are predefined.
- Categories cannot be created, updated or deleted in Version 1.
- Every AI prediction references exactly one category.
- Collection schedules reference these categories.

---

# Authorization Summary

| Endpoint | Citizen | Worker | Admin |
|-----------|---------|--------|-------|
|GET /categories|✅|✅|✅|




# ============================================
# Feature 6 - Collection Schedule
# ============================================

## Overview

The Collection Schedule module provides waste collection schedules for each ward.

Schedules help citizens know when specific categories of waste will be collected and help workers plan daily operations.

Schedules are managed by administrators.

---

# Permissions

| Role | Access |
|------|--------|
| Citizen | View |
| Worker | View |
| Admin | View & Manage |

---

# ============================================
# GET /schedules
# ============================================

## Purpose

Returns all collection schedules.

Supports optional filtering by ward and waste category.

---

## Authentication

Citizen

Worker

Admin

---

## Query Parameters

| Parameter | Description |
|-----------|-------------|
| wardId | Filter by ward |
| categoryId | Filter by waste category |
| day | Filter by day of week |

Example

GET /schedules?wardId=UUID&day=Monday

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "data": [
    {
      "scheduleId": "UUID",
      "ward": "Ward 12",
      "category": "Plastic",
      "day": "Monday",
      "startTime": "08:00",
      "endTime": "12:00"
    }
  ]
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
|401|Unauthorized|
|404|Schedule Not Found|

---

# ============================================
# GET /wards/{id}/schedule
# ============================================

## Purpose

Returns the complete collection schedule for a specific ward.

---

## Authentication

Citizen

Worker

Admin

---

## Path Parameters

| Parameter | Type |
|-----------|------|
| id | UUID |

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "data": {
    "wardId": "UUID",
    "wardName": "Ward 12",
    "schedule": [
      {
        "category": "Plastic",
        "day": "Monday",
        "startTime": "08:00",
        "endTime": "12:00"
      },
      {
        "category": "Organic",
        "day": "Wednesday",
        "startTime": "07:00",
        "endTime": "11:00"
      }
    ]
  }
}
```

---

# ============================================
# POST /schedules
# ============================================

## Purpose

Create a new collection schedule.

---

## Authentication

Admin

---

## Request Body

```json
{
  "wardId": "UUID",
  "categoryId": "UUID",
  "day": "Monday",
  "startTime": "08:00",
  "endTime": "12:00"
}
```

---

## Success Response

**201 Created**

```json
{
  "success": true,
  "message": "Collection schedule created successfully."
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
|400|Validation Failed|
|404|Ward Not Found|
|404|Category Not Found|
|409|Duplicate Schedule|

---

# ============================================
# PATCH /schedules/{id}
# ============================================

## Purpose

Update an existing collection schedule.

---

## Authentication

Admin

---

## Request Body

```json
{
  "day": "Tuesday",
  "startTime": "09:00",
  "endTime": "01:00"
}
```

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "message": "Collection schedule updated successfully."
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
|400|Validation Failed|
|404|Schedule Not Found|

---

# ============================================
# DELETE /schedules/{id}
# ============================================

## Purpose

Remove a collection schedule.

---

## Authentication

Admin

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "message": "Collection schedule deleted successfully."
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
|404|Schedule Not Found|

---

# Business Rules

- Every schedule belongs to one ward.
- Every schedule belongs to one waste category.
- A ward can have multiple schedules.
- The same waste category can appear in multiple wards.
- Duplicate schedules (same ward, category and day) are not allowed.
- Citizens and workers have read-only access.
- Only administrators may create, update or delete schedules.

---

# Authorization Summary

| Endpoint | Citizen | Worker | Admin |
|-----------|---------|--------|-------|
|GET /schedules|✅|✅|✅|
|GET /wards/{id}/schedule|✅|✅|✅|
|POST /schedules|❌|❌|✅|
|PATCH /schedules/{id}|❌|❌|✅|
|DELETE /schedules/{id}|❌|❌|✅|




# ============================================
# Feature 7 - Dashboard
# ============================================

## Overview

The Dashboard module provides a personalized overview based on the authenticated user's role.

Different dashboards are available for:

- Citizen
- Worker
- Admin

Dashboard data is read-only.

No dashboard endpoint modifies system data.

---

# Permissions

| Role | Dashboard |
|------|-----------|
| Citizen | Citizen Dashboard |
| Worker | Worker Dashboard |
| Admin | Admin Dashboard |

---

# ============================================
# GET /dashboard/citizen
# ============================================

## Purpose

Returns a summary of the logged-in citizen's waste reports and household information.

---

## Authentication

Citizen

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "data": {
    "citizen": {
      "id": "UUID",
      "name": "Sameer"
    },

    "household": {
      "id": "UUID",
      "ward": "Ward 12"
    },

    "summary": {
      "totalReports": 18,
      "pendingReports": 2,
      "completedReports": 16
    },

    "recentReports": [
      {
        "reportId": "UUID",
        "status": "PENDING",
        "category": "Plastic",
        "createdAt": "2026-07-27T10:30:00Z"
      }
    ],

    "nextCollection": {
      "category": "Organic",
      "day": "Wednesday",
      "startTime": "07:00",
      "endTime": "11:00"
    }
  }
}
```

---

# ============================================
# GET /dashboard/worker
# ============================================

## Purpose

Returns a summary of the logged-in worker's workload.

---

## Authentication

Worker

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "data": {

    "assignedWards": [
      "Ward 12",
      "Ward 15"
    ],

    "summary": {
      "pendingReports": 12,
      "acceptedReports": 4,
      "inProgressReports": 2,
      "completedToday": 8
    },

    "activeReports": [
      {
        "reportId": "UUID",
        "status": "IN_PROGRESS",
        "address": "MG Road"
      }
    ]
  }
}
```

---

# ============================================
# GET /dashboard/admin
# ============================================

## Purpose

Returns municipality-wide operational statistics.

---

## Authentication

Admin

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "data": {

    "summary": {

      "totalCitizens": 520,

      "totalWorkers": 18,

      "totalReports": 340,

      "pendingReports": 39,

      "completedReports": 301
    },

    "wardOverview": [

      {
        "ward": "Ward 12",

        "reports": 52,

        "pending": 4
      }

    ],

    "workerOverview": [

      {
        "worker": "John",

        "activeReports": 6,

        "completedToday": 8
      }

    ]
  }
}
```

---

# Error Responses

| Status | Description |
|--------|-------------|
|401|Unauthorized|
|403|Forbidden|

---

# Business Rules

- Dashboard data is generated dynamically.
- Users may only access their own dashboard.
- Dashboard endpoints are read-only.
- Dashboard summaries are derived from existing system data.
- Dashboard does not store separate statistics.

---

# Authorization Summary

| Endpoint | Citizen | Worker | Admin |
|-----------|---------|--------|-------|
|GET /dashboard/citizen|✅|❌|❌|
|GET /dashboard/worker|❌|✅|❌|
|GET /dashboard/admin|❌|❌|✅|




# ============================================
# Feature 8 - Predictions
# ============================================

## Overview

The Predictions module provides AI-generated waste classification results.

Predictions are generated automatically during waste report submission using the Edge AI model.

This module is read-only.

Prediction creation and updates are handled internally by the backend.

---

# Permissions

| Role | Access |
|------|--------|
| Citizen | View Own Predictions |
| Worker | View Predictions for Accessible Reports |
| Admin | View All Predictions |

---

# ============================================
# GET /predictions/{reportId}
# ============================================

## Purpose

Returns the AI prediction associated with a waste report.

---

## Authentication

Citizen

Worker

Admin

---

## Authorization Rules

Citizen

Can only view predictions for their own reports.


Worker

Can only view predictions for reports in their assigned ward(s).


Admin

Can view all predictions.

---

## Path Parameters

| Parameter | Type |
|-----------|------|
| reportId | UUID |

---

## Success Response

**200 OK**

```json
{
  "success": true,
  "data": {
    "reportId": "UUID",

    "category": "Plastic",

    "confidence": 94.20,

    "model": "MobileNet",

    "predictedAt": "2026-07-27T10:30:00Z"
  }
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
|401|Unauthorized|
|403|Forbidden|
|404|Prediction Not Found|

---

# Prediction Workflow

Citizen Uploads Image

↓

Edge AI (MobileNet)

↓

Prediction Generated

↓

Prediction Stored

↓

Returned to Citizen

↓

Available through GET /predictions/{reportId}

---

# Business Rules

- Every waste report has exactly one prediction.
- Predictions are generated automatically.
- Predictions cannot be modified.
- Predictions cannot be deleted.
- Prediction confidence is stored as a percentage.
- Model name is stored for traceability.

---

# Authorization Summary

| Endpoint | Citizen | Worker | Admin |
|-----------|---------|--------|-------|
|GET /predictions/{reportId}|Own|Ward Reports|All|

---

# Progress

| Feature | Status |
|----------|--------|
| Authentication | ✅ Completed |
| Ward & Household | ✅ Completed |
| Waste Reporting | ✅ Completed |
| Workforce Management | ✅ Completed |
| Categories | ✅ Completed |
| Collection Schedule | ✅ Completed |
| Dashboard | ✅ Completed |
| Predictions | ✅ Completed |
