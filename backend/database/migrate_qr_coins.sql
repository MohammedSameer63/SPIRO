-- ============================================================
-- SPIRO migration: QR pickup validations + green coins
-- Run this ONCE on databases created before these features existed
-- (i.e. if you see: relation "pickup_validations" does not exist).
--
-- Run as a privileged user (table owner / superuser), e.g.:
--   sudo -u postgres psql -d spiro -f backend/database/migrate_qr_coins.sql
-- or:
--   psql "$DATABASE_URL" -f backend/database/migrate_qr_coins.sql
-- (the second form only works if your user owns the tables / can CREATE)
-- ============================================================

CREATE TABLE IF NOT EXISTS pickup_validations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    report_id UUID UNIQUE NOT NULL,

    household_id UUID NOT NULL,

    worker_id UUID NOT NULL,

    method VARCHAR(20) NOT NULL DEFAULT 'QR_SCAN',

    validated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (report_id)
        REFERENCES waste_reports(id)
        ON DELETE CASCADE,

    FOREIGN KEY (household_id)
        REFERENCES households(id)
        ON DELETE RESTRICT,

    FOREIGN KEY (worker_id)
        REFERENCES users(id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS coin_wallets (
    user_id UUID PRIMARY KEY,

    balance INTEGER NOT NULL DEFAULT 0,

    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS coin_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL,

    amount INTEGER NOT NULL,

    reason VARCHAR(30) NOT NULL,

    report_id UUID,

    note TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE,

    FOREIGN KEY (report_id)
        REFERENCES waste_reports(id)
        ON DELETE SET NULL
);

-- At most one completion award per report (NULL report_ids stay distinct).
DROP INDEX IF EXISTS uq_coin_tx_completion_report;
CREATE UNIQUE INDEX uq_coin_tx_completion_report
ON coin_transactions(report_id)
WHERE reason = 'completion' AND report_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_coin_tx_user
ON coin_transactions(user_id);

CREATE INDEX IF NOT EXISTS idx_pickup_validations_report
ON pickup_validations(report_id);
