# SPIRO REST API Documentation

**Version:** 1.0 (MVP)

**Status:** Draft

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
4. Assignments
5. Categories
6. Collection Schedule
7. Dashboard
8. Predictions

---

# Feature 1 — Authentication

Authentication handles user registration, login, logout, and retrieval of the currently authenticated user's profile.

---

## POST `/auth/register`

### Purpose

Register a new citizen account and create a household.

### Authentication Required

**No**

---

### Request Body

```json
{
  "name": "Sameer",
  "email": "sameer@example.com",
  "password": "Password@123",
  "phone": "9876543210",

  "wardId": "UUID",
  "houseNumber": "12A",
  "streetName": "MG Road",
  "address": "12A MG Road, Bengaluru"
}
```

---

### Validation Rules

#### User

| Field | Rules |
|-------|-------|
| name | Required, 2–100 characters |
| email | Required, valid email format, unique |
| password | Minimum 8 characters |
| password | At least one uppercase letter |
| password | At least one lowercase letter |
| password | At least one number |
| password | At least one special character |
| phone | Optional, valid phone number format |

#### Household

| Field | Rules |
|-------|-------|
| wardId | Required, must exist |
| houseNumber | Required |
| address | Required |
| streetName | Optional |

Additional Business Rules:

- Selected ward must exist.
- Duplicate households (same ward + house number + address) are not allowed.
- Household is created automatically during registration.
- Citizen is automatically linked to the newly created household.

---

### Success Response

**201 Created**

```json
{
  "success": true,
  "message": "Registration successful.",
  "data": {
    "user": {
      "id": "UUID",
      "name": "Sameer",
      "role": "CITIZEN"
    },
    "token": "JWT_TOKEN"
  }
}
```

---

### Error Responses

| Status | Description |
|--------|-------------|
| 400 | Validation failed |
| 404 | Ward not found |
| 409 | Email already exists |
| 409 | Household already exists |
| 500 | Internal server error |

---

### Notes

- Email must be unique.
- Password is stored as a hashed value.
- New users are registered with the **CITIZEN** role.
- Household QR code is generated automatically after successful registration.

---

## POST `/auth/login`

### Purpose

Authenticate an existing user and return a JWT access token.

### Authentication Required

**No**

---

### Request Body

```json
{
    "email": "john@example.com",
    "password": "Password@123"
}
```

---

### Validation Rules

| Field | Rules |
|------|-------|
| email | Required |
| password | Required |

---

### Success Response

**200 OK**

```json
{
    "success": true,
    "message": "Login successful.",
    "data": {
        "token": "JWT_TOKEN",
        "user": {
            "id": "UUID",
            "name": "John Doe",
            "role": "CITIZEN"
        }
    }
}
```

---

### Error Responses

| Status | Description |
|---------|-------------|
| 401 | Invalid credentials |
| 403 | Account disabled |

---

### Notes

- JWT expires after **24 hours**.
- JWT must be included in the `Authorization` header for all protected endpoints.

---

## POST `/auth/logout`

### Purpose

Logout the currently authenticated user.

### Authentication Required

**No**

### Allowed Roles

Citizen • Worker • Admin

---

### Request Body

None

---

### Success Response

**200 OK**

```json
{
    "success": true,
    "message": "Logged out successfully."
}
```

---

### Error Responses

| Status | Description |
|---------|-------------|
| 401 | Unauthorized |

---

### Notes

- In JWT authentication, logout is typically handled on the client by removing the stored token.
- If refresh tokens or token blacklisting are introduced later, this endpoint can invalidate the active session.

---

## GET `/auth/me`

### Purpose

Retrieve the profile of the currently authenticated user.

### Authentication Required

**No**

### Allowed Roles

Citizen • Worker • Admin

---

### Request Body

None

---

### Success Response

**200 OK**

```json
{
    "success": true,
    "data": {
        "id": "UUID",
        "name": "John Doe",
        "email": "john@example.com",
        "role": "CITIZEN"
    }
}
```

---

### Error Responses

| Status | Description |
|---------|-------------|
| 401 | Unauthorized |

---

### Notes

The user is identified using the JWT access token.

---

# Authentication Flow

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
Authorization: Bearer <token>
      │
      ▼
Access Protected APIs
```

---

# Authentication Rules

## Password

Minimum Requirements

- Minimum 8 characters
- One uppercase letter
- One lowercase letter
- One numeric digit
- One special character

---

## Email

- Must be unique
- Must be a valid email address

---

## Phone

- Optional
- Must be a valid mobile number if provided

---

## JWT

- Expiry: 24 Hours
- Type: Bearer Token

---

## Authorization Header

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
| latitude | Decimal | Yes |
| longitude | Decimal | Yes |
| address | String | Yes |

---

## Validation Rules

Image

- JPG
- JPEG
- PNG

Maximum size

10 MB

Location

Latitude

-90 → +90

Longitude

-180 → +180

Address

Required

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

# Overall Business Rules ig

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
