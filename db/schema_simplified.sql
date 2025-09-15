-- Simplified schema for AI Wellness Assistant
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Enable RLS for all tables
SET row_security = on;

-- Create auth schema
CREATE SCHEMA IF NOT EXISTS auth;

-- Core patient table
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(20),
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    medical_record_number VARCHAR(50) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Medical conditions
CREATE TABLE IF NOT EXISTS conditions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    condition_code VARCHAR(20) NOT NULL,
    condition_name VARCHAR(255) NOT NULL,
    severity VARCHAR(20),
    diagnosis_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Vital signs
CREATE TABLE IF NOT EXISTS vitals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    measurement_date TIMESTAMP WITH TIME ZONE NOT NULL,
    systolic_bp INTEGER,
    diastolic_bp INTEGER,
    heart_rate INTEGER,
    weight_kg DECIMAL(5,2),
    height_cm DECIMAL(5,2),
    bmi DECIMAL(5,2),
    hba1c DECIMAL(4,2),
    glucose_mg_dl INTEGER,
    cholesterol_total INTEGER,
    cholesterol_hdl INTEGER,
    cholesterol_ldl INTEGER,
    triglycerides INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Medications
CREATE TABLE IF NOT EXISTS medications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    medication_name VARCHAR(255) NOT NULL,
    dosage VARCHAR(100),
    frequency VARCHAR(100),
    route VARCHAR(50),
    prescribed_date DATE,
    start_date DATE,
    end_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    prescriber_name VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Recommendations
CREATE TABLE IF NOT EXISTS recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    category VARCHAR(100),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium',
    confidence_score DECIMAL(3,2),
    status VARCHAR(20) DEFAULT 'pending',
    created_by VARCHAR(100),
    reviewed_by UUID,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    evidence_summary TEXT,
    contraindications TEXT,
    follow_up_needed BOOLEAN DEFAULT FALSE,
    follow_up_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Evidence links
CREATE TABLE IF NOT EXISTS evidence_links (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recommendation_id UUID NOT NULL REFERENCES recommendations(id) ON DELETE CASCADE,
    source_type VARCHAR(50) NOT NULL,
    source_id VARCHAR(255),
    url TEXT,
    title VARCHAR(500),
    confidence_score DECIMAL(3,2),
    relevance_score DECIMAL(3,2),
    quality_rating VARCHAR(20),
    publication_date DATE,
    abstract TEXT,
    key_findings TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User roles for auth
CREATE TABLE IF NOT EXISTS user_roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL UNIQUE,
    role VARCHAR(50) NOT NULL DEFAULT 'patient',
    permissions JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Clinician-patient relationships
CREATE TABLE IF NOT EXISTS clinician_patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clinician_id UUID NOT NULL,
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) DEFAULT 'primary_care',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(clinician_id, patient_id)
);

-- Audit trail
CREATE TABLE IF NOT EXISTS audit_trail (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_patients_user_id ON patients(user_id);
CREATE INDEX IF NOT EXISTS idx_patients_email ON patients(email);
CREATE INDEX IF NOT EXISTS idx_conditions_patient_id ON conditions(patient_id);
CREATE INDEX IF NOT EXISTS idx_vitals_patient_id ON vitals(patient_id);
CREATE INDEX IF NOT EXISTS idx_vitals_measurement_date ON vitals(measurement_date);
CREATE INDEX IF NOT EXISTS idx_medications_patient_id ON medications(patient_id);
CREATE INDEX IF NOT EXISTS idx_medications_status ON medications(status);
CREATE INDEX IF NOT EXISTS idx_recommendations_patient_id ON recommendations(patient_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status);
CREATE INDEX IF NOT EXISTS idx_evidence_links_recommendation_id ON evidence_links(recommendation_id);
CREATE INDEX IF NOT EXISTS idx_clinician_patients_clinician_id ON clinician_patients(clinician_id);
CREATE INDEX IF NOT EXISTS idx_clinician_patients_patient_id ON clinician_patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_trail_user_id ON audit_trail(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_trail_created_at ON audit_trail(created_at);

-- Insert some sample data
INSERT INTO patients (first_name, last_name, date_of_birth, gender, email, medical_record_number)
VALUES 
    ('John', 'Doe', '1975-06-15', 'Male', 'john.doe@example.com', 'MRN001'),
    ('Jane', 'Smith', '1982-03-22', 'Female', 'jane.smith@example.com', 'MRN002'),
    ('Robert', 'Johnson', '1968-09-30', 'Male', 'robert.johnson@example.com', 'MRN003')
ON CONFLICT (email) DO NOTHING;

-- Insert sample conditions
INSERT INTO conditions (patient_id, condition_code, condition_name, severity, diagnosis_date)
SELECT 
    p.id,
    'E11',
    'Type 2 Diabetes Mellitus',
    'moderate',
    '2020-01-15'::date
FROM patients p WHERE p.email = 'john.doe@example.com'
ON CONFLICT DO NOTHING;

INSERT INTO conditions (patient_id, condition_code, condition_name, severity, diagnosis_date)
SELECT 
    p.id,
    'I10',
    'Essential Hypertension',
    'mild',
    '2019-06-10'::date
FROM patients p WHERE p.email = 'jane.smith@example.com'
ON CONFLICT DO NOTHING;

-- Insert sample vitals
INSERT INTO vitals (patient_id, measurement_date, systolic_bp, diastolic_bp, heart_rate, weight_kg, height_cm, hba1c, glucose_mg_dl)
SELECT 
    p.id,
    NOW() - INTERVAL '1 day',
    140,
    90,
    72,
    85.5,
    175.0,
    7.2,
    150
FROM patients p WHERE p.email = 'john.doe@example.com';

INSERT INTO vitals (patient_id, measurement_date, systolic_bp, diastolic_bp, heart_rate, weight_kg, height_cm)
SELECT 
    p.id,
    NOW() - INTERVAL '2 days',
    150,
    95,
    78,
    68.0,
    162.0
FROM patients p WHERE p.email = 'jane.smith@example.com';
