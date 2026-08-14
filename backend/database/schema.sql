-- ============================================================
-- SPIRO - Version 1 Database Schema
-- PostgreSQL
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- ENUMS
-- ============================================================

CREATE TYPE user_role AS ENUM (
    'CITIZEN',
    'WORKER',
    'ADMIN'
);

CREATE TYPE user_status AS ENUM (
    'ACTIVE',
    'INACTIVE',
    'SUSPENDED'
);

CREATE TYPE report_status AS ENUM (
    'PENDING',
    'ACCEPTED',
    'IN_PROGRESS',
    'COMPLETED'
);

CREATE TYPE audit_event AS ENUM (
    'USER_CREATED',
    'LOGIN',
    'REPORT_CREATED',
    'REPORT_UPDATED',
    'REPORT_ASSIGNED',
    'REPORT_COMPLETED'
);

-- ============================================================
-- WARDS
-- ============================================================

CREATE TABLE wards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    name VARCHAR(100) NOT NULL UNIQUE,
    zone VARCHAR(100),
    description TEXT
);

-- ============================================================
-- HOUSEHOLDS
-- ============================================================

CREATE TABLE households (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    ward_id UUID NOT NULL,

    house_number VARCHAR(20) NOT NULL,

    street_name VARCHAR(255) NOT NULL,

    address TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_households_ward
        FOREIGN KEY (ward_id)
        REFERENCES wards(id)
        ON DELETE RESTRICT,

    CONSTRAINT uq_household_location
        UNIQUE (ward_id, house_number, street_name)
);

-- ============================================================
-- USERS
-- ============================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    household_id UUID,

    name VARCHAR(100) NOT NULL,

    email VARCHAR(255) UNIQUE NOT NULL,

    phone VARCHAR(15),

    password_hash VARCHAR(255) NOT NULL,

    role user_role NOT NULL DEFAULT 'CITIZEN',

    status user_status NOT NULL DEFAULT 'ACTIVE',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (household_id)
        REFERENCES households(id)
        ON DELETE SET NULL
);

-- ============================================================
-- WASTE CATEGORIES
-- ============================================================

CREATE TABLE waste_categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    name VARCHAR(50) UNIQUE NOT NULL,

    description TEXT
);

-- ============================================================
-- COLLECTION SCHEDULES
-- ============================================================

CREATE TABLE collection_schedules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    ward_id UUID NOT NULL,

    waste_category_id UUID NOT NULL,

    day_of_week VARCHAR(20) NOT NULL,

    start_time TIME,

    end_time TIME,

    FOREIGN KEY (ward_id)
        REFERENCES wards(id)
        ON DELETE RESTRICT,

    FOREIGN KEY (waste_category_id)
        REFERENCES waste_categories(id)
        ON DELETE RESTRICT
);

-- ============================================================
-- WASTE REPORTS
-- ============================================================

CREATE TABLE waste_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    user_id UUID NOT NULL,

    schedule_id UUID,

    description TEXT,

    image_url TEXT NOT NULL,

    latitude DECIMAL(9,6),

    longitude DECIMAL(9,6),

    address TEXT,

    status report_status NOT NULL DEFAULT 'PENDING',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE RESTRICT,

    FOREIGN KEY (schedule_id)
        REFERENCES collection_schedules(id)
        ON DELETE SET NULL
);

-- ============================================================
-- PREDICTIONS
-- ============================================================

CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    report_id UUID UNIQUE NOT NULL,

    category_id UUID NOT NULL,

    confidence DECIMAL(5,2)
        CHECK (confidence >= 0 AND confidence <= 100),

    model_used VARCHAR(50),

    prediction_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (report_id)
        REFERENCES waste_reports(id)
        ON DELETE CASCADE,

    FOREIGN KEY (category_id)
        REFERENCES waste_categories(id)
        ON DELETE RESTRICT
);

-- ============================================================
-- ASSIGNMENTS
-- ============================================================

CREATE TABLE assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    report_id UUID UNIQUE NOT NULL,

    worker_id UUID NOT NULL,

    assigned_by UUID NOT NULL,

    assigned_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    completed_at TIMESTAMPTZ,

    FOREIGN KEY (report_id)
        REFERENCES waste_reports(id)
        ON DELETE CASCADE,

    FOREIGN KEY (worker_id)
        REFERENCES users(id)
        ON DELETE RESTRICT,

    FOREIGN KEY (assigned_by)
        REFERENCES users(id)
        ON DELETE RESTRICT
);

-- ============================================================
-- AUDIT LOGS
-- ============================================================

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    event_type audit_event NOT NULL,

    entity_type VARCHAR(50),

    entity_id UUID,

    performed_by UUID,

    metadata JSONB NOT NULL DEFAULT '{}',

    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (performed_by)
        REFERENCES users(id)
        ON DELETE SET NULL
);


-- ============================================================
-- WORKER - WARD MAPPING
-- ============================================================

CREATE TABLE worker_wards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    worker_id UUID NOT NULL,

    ward_id UUID NOT NULL,

    assigned_by UUID NOT NULL,

    assigned_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (worker_id)
        REFERENCES users(id)
        ON DELETE CASCADE,

    FOREIGN KEY (ward_id)
        REFERENCES wards(id)
        ON DELETE CASCADE,

    FOREIGN KEY (assigned_by)
        REFERENCES users(id)
        ON DELETE RESTRICT,

    CONSTRAINT unique_worker_ward
        UNIQUE (worker_id, ward_id)
);

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX idx_worker_wards_worker
ON worker_wards(worker_id);

CREATE INDEX idx_worker_wards_ward
ON worker_wards(ward_id);

CREATE INDEX idx_users_email
ON users(email);

CREATE INDEX idx_households_ward
ON households(ward_id);

CREATE INDEX idx_reports_user
ON waste_reports(user_id);

CREATE INDEX idx_reports_status
ON waste_reports(status);

CREATE INDEX idx_reports_created_at
ON waste_reports(created_at);

CREATE INDEX idx_assignments_worker
ON assignments(worker_id);

CREATE INDEX idx_predictions_report
ON predictions(report_id);

CREATE INDEX idx_audit_timestamp
ON audit_logs(timestamp);
