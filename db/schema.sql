-- AI Wellness Assistant Database Schema
-- Supabase/PostgreSQL with Row Level Security (RLS)

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Core domain tables

-- Patients table (links to auth.users)
CREATE TABLE patients (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
    demographics jsonb NOT NULL DEFAULT '{}',
    consent jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Patient vitals and measurements
CREATE TABLE vitals (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id uuid REFERENCES patients(id) ON DELETE CASCADE,
    ts timestamptz NOT NULL,
    type text NOT NULL,              -- glucose, sbp, dbp, hr, weight, bmi
    value numeric NOT NULL,
    unit text NOT NULL,              -- canonical units after normalization
    source text DEFAULT 'manual',   -- manual, device, ehr
    created_at timestamptz DEFAULT now(),
    
    -- Create index for time-series queries
    CONSTRAINT vitals_ts_idx UNIQUE (patient_id, type, ts)
);

-- Patient medications
CREATE TABLE medications (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id uuid REFERENCES patients(id) ON DELETE CASCADE,
    rxnorm_code text,                -- normalized RxNorm identifier
    name text NOT NULL,
    dosage text,
    schedule jsonb DEFAULT '{}',     -- structured dosing information
    active boolean DEFAULT true,
    start_date date,
    end_date date,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Medical conditions/diagnoses
CREATE TABLE conditions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id uuid REFERENCES patients(id) ON DELETE CASCADE,
    icd10_code text,
    name text NOT NULL,
    severity text,                   -- mild, moderate, severe
    onset_date date,
    active boolean DEFAULT true,
    created_at timestamptz DEFAULT now()
);

-- Care plan recommendations
CREATE TABLE recommendations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id uuid REFERENCES patients(id) ON DELETE CASCADE,
    snapshot_ts timestamptz NOT NULL,
    careplan jsonb NOT NULL DEFAULT '{}',         -- structured recommendations by category
    explainer jsonb DEFAULT '{}',                 -- SHAP summary, NL explanation (patient-safe)
    model_version text,
    status text DEFAULT 'pending',                -- pending/approved/rejected/flagged
    risk_score numeric,                           -- primary risk probability
    risk_category text,                           -- low/medium/high
    created_by_user_id uuid REFERENCES auth.users(id),
    approved_by_user_id uuid REFERENCES auth.users(id),
    approved_at timestamptz,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    
    CONSTRAINT valid_status CHECK (status IN ('pending', 'approved', 'rejected', 'flagged'))
);

-- Evidence supporting recommendations
CREATE TABLE evidence_links (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_id uuid REFERENCES recommendations(id) ON DELETE CASCADE,
    source_type text NOT NULL,                    -- guideline/pubmed/openfda/rxnorm
    url text,
    title text,
    weight numeric DEFAULT 0.0,                  -- scoring per PRD rubric
    snippet text,
    metadata jsonb DEFAULT '{}',                  -- source-specific metadata
    created_at timestamptz DEFAULT now(),
    
    CONSTRAINT valid_source_type CHECK (source_type IN ('guideline', 'pubmed', 'openfda', 'rxnorm', 'ada'))
);

-- Comprehensive audit trail
CREATE TABLE audit_logs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_type text NOT NULL,                    -- system/clinician/patient
    actor_id uuid,                               -- references auth.users(id)
    event text NOT NULL,                         -- recommendation.created/approved/overridden
    entity_type text,                            -- patient, recommendation, etc.
    entity_id uuid,
    payload jsonb DEFAULT '{}',
    payload_hash text,                           -- immutable snapshot verification
    model_version text,
    ts timestamptz DEFAULT now(),
    
    CONSTRAINT valid_actor_type CHECK (actor_type IN ('system', 'clinician', 'patient', 'admin'))
);

-- ML model version tracking
CREATE TABLE model_versions (
    id serial PRIMARY KEY,
    tag text UNIQUE NOT NULL,                   -- e.g., risk_lgbm_v0.1
    algorithm text NOT NULL,                    -- lightgbm, xgboost, etc.
    training_meta jsonb DEFAULT '{}',           -- hyperparams, dataset info, metrics
    model_path text,                            -- file system path to serialized model
    performance_metrics jsonb DEFAULT '{}',     -- AUPRC, ROC, calibration metrics
    created_at timestamptz DEFAULT now(),
    is_active boolean DEFAULT false             -- current production model
);

-- Clinician-patient assignments for RBAC
CREATE TABLE clinician_patients (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    clinician_user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
    patient_id uuid REFERENCES patients(id) ON DELETE CASCADE,
    assigned_at timestamptz DEFAULT now(),
    active boolean DEFAULT true,
    
    UNIQUE(clinician_user_id, patient_id)
);

-- User roles for RBAC
CREATE TABLE user_roles (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
    role text NOT NULL,                         -- patient, clinician, admin
    metadata jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    
    CONSTRAINT valid_role CHECK (role IN ('patient', 'clinician', 'admin')),
    UNIQUE(user_id, role)
);

-- Create indexes for performance
CREATE INDEX idx_vitals_patient_type_ts ON vitals(patient_id, type, ts DESC);
CREATE INDEX idx_medications_patient_active ON medications(patient_id, active);
CREATE INDEX idx_recommendations_patient_status ON recommendations(patient_id, status, created_at DESC);
CREATE INDEX idx_evidence_links_recommendation ON evidence_links(recommendation_id);
CREATE INDEX idx_audit_logs_entity ON audit_logs(entity_type, entity_id, ts DESC);
CREATE INDEX idx_clinician_patients_active ON clinician_patients(clinician_user_id, active);

-- Enable Row Level Security (RLS)
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE vitals ENABLE ROW LEVEL SECURITY;
ALTER TABLE medications ENABLE ROW LEVEL SECURITY;
ALTER TABLE conditions ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE clinician_patients ENABLE ROW LEVEL SECURITY;

-- RLS Policies

-- Patients can only see their own data
CREATE POLICY "Patients can view own data" ON patients
    FOR ALL USING (auth.uid() = user_id);

-- Clinicians can see assigned patients
CREATE POLICY "Clinicians can view assigned patients" ON patients
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM clinician_patients cp
            JOIN user_roles ur ON ur.user_id = cp.clinician_user_id
            WHERE cp.patient_id = patients.id
            AND cp.clinician_user_id = auth.uid()
            AND cp.active = true
            AND ur.role = 'clinician'
        )
    );

-- Admins can see all
CREATE POLICY "Admins can view all patients" ON patients
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM user_roles
            WHERE user_id = auth.uid() AND role = 'admin'
        )
    );

-- Similar policies for related tables
CREATE POLICY "Patient data access" ON vitals
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM patients p
            WHERE p.id = vitals.patient_id
            AND (
                p.user_id = auth.uid() OR -- Patient owns data
                EXISTS ( -- Clinician assigned
                    SELECT 1 FROM clinician_patients cp
                    JOIN user_roles ur ON ur.user_id = cp.clinician_user_id
                    WHERE cp.patient_id = p.id
                    AND cp.clinician_user_id = auth.uid()
                    AND cp.active = true
                    AND ur.role = 'clinician'
                ) OR
                EXISTS ( -- Admin access
                    SELECT 1 FROM user_roles
                    WHERE user_id = auth.uid() AND role = 'admin'
                )
            )
        )
    );

-- Apply similar RLS pattern to other tables
CREATE POLICY "Patient data access" ON medications
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM patients p
            WHERE p.id = medications.patient_id
            AND (
                p.user_id = auth.uid() OR
                EXISTS (
                    SELECT 1 FROM clinician_patients cp
                    JOIN user_roles ur ON ur.user_id = cp.clinician_user_id
                    WHERE cp.patient_id = p.id
                    AND cp.clinician_user_id = auth.uid()
                    AND cp.active = true
                    AND ur.role = 'clinician'
                ) OR
                EXISTS (
                    SELECT 1 FROM user_roles
                    WHERE user_id = auth.uid() AND role = 'admin'
                )
            )
        )
    );

CREATE POLICY "Patient data access" ON conditions
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM patients p
            WHERE p.id = conditions.patient_id
            AND (
                p.user_id = auth.uid() OR
                EXISTS (
                    SELECT 1 FROM clinician_patients cp
                    JOIN user_roles ur ON ur.user_id = cp.clinician_user_id
                    WHERE cp.patient_id = p.id
                    AND cp.clinician_user_id = auth.uid()
                    AND cp.active = true
                    AND ur.role = 'clinician'
                ) OR
                EXISTS (
                    SELECT 1 FROM user_roles
                    WHERE user_id = auth.uid() AND role = 'admin'
                )
            )
        )
    );

CREATE POLICY "Patient data access" ON recommendations
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM patients p
            WHERE p.id = recommendations.patient_id
            AND (
                p.user_id = auth.uid() OR
                EXISTS (
                    SELECT 1 FROM clinician_patients cp
                    JOIN user_roles ur ON ur.user_id = cp.clinician_user_id
                    WHERE cp.patient_id = p.id
                    AND cp.clinician_user_id = auth.uid()
                    AND cp.active = true
                    AND ur.role = 'clinician'
                ) OR
                EXISTS (
                    SELECT 1 FROM user_roles
                    WHERE user_id = auth.uid() AND role = 'admin'
                )
            )
        )
    );

-- Evidence links inherit from recommendations
CREATE POLICY "Evidence access through recommendations" ON evidence_links
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM recommendations r
            JOIN patients p ON p.id = r.patient_id
            WHERE r.id = evidence_links.recommendation_id
            AND (
                p.user_id = auth.uid() OR
                EXISTS (
                    SELECT 1 FROM clinician_patients cp
                    JOIN user_roles ur ON ur.user_id = cp.clinician_user_id
                    WHERE cp.patient_id = p.id
                    AND cp.clinician_user_id = auth.uid()
                    AND cp.active = true
                    AND ur.role = 'clinician'
                ) OR
                EXISTS (
                    SELECT 1 FROM user_roles
                    WHERE user_id = auth.uid() AND role = 'admin'
                )
            )
        )
    );

-- Audit logs - restricted access
CREATE POLICY "Audit log access" ON audit_logs
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM user_roles
            WHERE user_id = auth.uid() AND role IN ('clinician', 'admin')
        )
    );

-- Clinician assignments - only clinicians and admins can see
CREATE POLICY "Clinician assignment access" ON clinician_patients
    FOR ALL USING (
        auth.uid() = clinician_user_id OR
        EXISTS (
            SELECT 1 FROM user_roles
            WHERE user_id = auth.uid() AND role = 'admin'
        )
    );

-- Functions for common operations

-- Function to get patient risk summary
CREATE OR REPLACE FUNCTION get_patient_risk_summary(p_patient_id uuid)
RETURNS jsonb AS $$
DECLARE
    result jsonb;
BEGIN
    SELECT jsonb_build_object(
        'latest_risk', (
            SELECT jsonb_build_object(
                'score', risk_score,
                'category', risk_category,
                'model_version', model_version,
                'created_at', created_at
            )
            FROM recommendations
            WHERE patient_id = p_patient_id
            AND status = 'approved'
            ORDER BY created_at DESC
            LIMIT 1
        ),
        'vital_trends', (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'type', type,
                    'latest_value', value,
                    'unit', unit,
                    'ts', ts
                )
            )
            FROM (
                SELECT DISTINCT ON (type) type, value, unit, ts
                FROM vitals
                WHERE patient_id = p_patient_id
                ORDER BY type, ts DESC
            ) latest_vitals
        ),
        'active_conditions', (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'name', name,
                    'icd10_code', icd10_code,
                    'severity', severity
                )
            )
            FROM conditions
            WHERE patient_id = p_patient_id AND active = true
        )
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to create audit log entry
CREATE OR REPLACE FUNCTION create_audit_log(
    p_actor_type text,
    p_actor_id uuid,
    p_event text,
    p_entity_type text,
    p_entity_id uuid,
    p_payload jsonb DEFAULT '{}'::jsonb,
    p_model_version text DEFAULT NULL
) RETURNS uuid AS $$
DECLARE
    log_id uuid;
    payload_hash_val text;
BEGIN
    -- Generate hash of payload for immutability verification
    payload_hash_val := encode(digest(p_payload::text, 'sha256'), 'hex');
    
    INSERT INTO audit_logs (
        actor_type, actor_id, event, entity_type, entity_id,
        payload, payload_hash, model_version
    ) VALUES (
        p_actor_type, p_actor_id, p_event, p_entity_type, p_entity_id,
        p_payload, payload_hash_val, p_model_version
    ) RETURNING id INTO log_id;
    
    RETURN log_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add update triggers
CREATE TRIGGER update_patients_updated_at BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_medications_updated_at BEFORE UPDATE ON medications
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_recommendations_updated_at BEFORE UPDATE ON recommendations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
