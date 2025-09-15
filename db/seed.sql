-- AI Wellness Assistant - Seed Data
-- Creates demo personas and synthetic patient cohort

-- Insert demo model version
INSERT INTO model_versions (tag, algorithm, training_meta, model_path, performance_metrics, is_active)
VALUES (
    'risk_lgbm_v0.1',
    'lightgbm',
    '{"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "dataset": "synthetic_cohort_v1"}',
    'services/ml_risk/models/risk_lgbm_v0_1.bin',
    '{"auprc": 0.78, "roc_auc": 0.82, "brier_score": 0.15, "calibration_slope": 0.95}',
    true
);

-- Create demo users (these would typically be created through Supabase Auth)
-- Note: In production, these IDs would come from auth.users table

-- Demo Patient: Maria Gonzalez (High Risk T2D)
INSERT INTO patients (id, user_id, demographics, consent)
VALUES (
    'f47ac10b-58cc-4372-a567-0e02b2c3d479',
    'f47ac10b-58cc-4372-a567-0e02b2c3d479',
    '{
        "name": "Maria Gonzalez",
        "age": 58,
        "sex": "F",
        "ethnicity": "Hispanic",
        "height_cm": 162,
        "weight_kg": 78.5,
        "bmi": 29.9,
        "family_history": {
            "diabetes": true,
            "heart_disease": true,
            "hypertension": true
        },
        "lifestyle": {
            "smoking": "never",
            "alcohol": "occasional",
            "exercise_mins_week": 60
        }
    }',
    '{"ehr_access": true, "data_sharing": true, "research_participation": false}'
);

-- Demo Clinician: Dr. Sarah Patel
INSERT INTO patients (id, user_id, demographics, consent)
VALUES (
    'c47ac10b-58cc-4372-a567-0e02b2c3d480',
    'c47ac10b-58cc-4372-a567-0e02b2c3d480',
    '{
        "name": "Dr. Sarah Patel",
        "role": "clinician",
        "specialty": "Endocrinology",
        "license_number": "MD12345",
        "institution": "City Medical Center"
    }',
    '{}'
);

-- User roles
INSERT INTO user_roles (user_id, role, metadata)
VALUES 
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', 'patient', '{}'),
    ('c47ac10b-58cc-4372-a567-0e02b2c3d480', 'clinician', '{"specialty": "endocrinology", "years_experience": 12}');

-- Clinician-patient assignment
INSERT INTO clinician_patients (clinician_user_id, patient_id)
VALUES ('c47ac10b-58cc-4372-a567-0e02b2c3d480', 'f47ac10b-58cc-4372-a567-0e02b2c3d479');

-- Maria's medical conditions
INSERT INTO conditions (patient_id, icd10_code, name, severity, onset_date, active)
VALUES 
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', 'E11.9', 'Type 2 Diabetes Mellitus', 'moderate', '2020-03-15', true),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', 'I10', 'Essential Hypertension', 'mild', '2019-08-22', true),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', 'E78.5', 'Hyperlipidemia', 'mild', '2021-01-10', true);

-- Maria's current medications
INSERT INTO medications (patient_id, rxnorm_code, name, dosage, schedule, active, start_date)
VALUES 
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '6809', 'Metformin', '500mg', '{"frequency": "twice_daily", "times": ["08:00", "20:00"], "with_food": true}', true, '2020-03-15'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '38454', 'Lisinopril', '10mg', '{"frequency": "once_daily", "times": ["08:00"]}', true, '2019-08-22'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '36567', 'Atorvastatin', '20mg', '{"frequency": "once_daily", "times": ["20:00"]}', true, '2021-01-10');

-- Maria's vital signs over the past 6 months (realistic progression)
INSERT INTO vitals (patient_id, ts, type, value, unit, source)
VALUES 
    -- Glucose readings (mg/dL) - showing some control issues
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-09-01 08:00:00+00', 'glucose_fasting', 148, 'mg/dL', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-08-25 08:15:00+00', 'glucose_fasting', 142, 'mg/dL', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-08-15 08:10:00+00', 'glucose_fasting', 156, 'mg/dL', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-08-01 08:05:00+00', 'glucose_fasting', 139, 'mg/dL', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-07-15 08:20:00+00', 'glucose_fasting', 145, 'mg/dL', 'device'),
    
    -- HbA1c (%) - quarterly readings
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-09-01 10:00:00+00', 'hba1c', 8.1, '%', 'lab'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-06-01 10:00:00+00', 'hba1c', 7.8, '%', 'lab'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-03-01 10:00:00+00', 'hba1c', 7.9, '%', 'lab'),
    
    -- Blood Pressure (mmHg) - showing mild hypertension
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-09-01 09:00:00+00', 'sbp', 138, 'mmHg', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-09-01 09:00:00+00', 'dbp', 88, 'mmHg', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-08-25 09:15:00+00', 'sbp', 142, 'mmHg', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-08-25 09:15:00+00', 'dbp', 92, 'mmHg', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-08-15 09:10:00+00', 'sbp', 135, 'mmHg', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-08-15 09:10:00+00', 'dbp', 85, 'mmHg', 'device'),
    
    -- Weight tracking (kg)
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-09-01 07:30:00+00', 'weight', 78.5, 'kg', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-08-15 07:30:00+00', 'weight', 79.2, 'kg', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-08-01 07:30:00+00', 'weight', 79.8, 'kg', 'device'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-07-15 07:30:00+00', 'weight', 80.1, 'kg', 'device'),
    
    -- Lipid panel (mg/dL) - recent labs
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-09-01 10:00:00+00', 'total_cholesterol', 198, 'mg/dL', 'lab'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-09-01 10:00:00+00', 'ldl_cholesterol', 118, 'mg/dL', 'lab'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-09-01 10:00:00+00', 'hdl_cholesterol', 42, 'mg/dL', 'lab'),
    ('f47ac10b-58cc-4372-a567-0e02b2c3d479', '2024-09-01 10:00:00+00', 'triglycerides', 185, 'mg/dL', 'lab');

-- Sample recommendation for Maria (high risk scenario)
INSERT INTO recommendations (
    id,
    patient_id,
    snapshot_ts,
    careplan,
    explainer,
    model_version,
    status,
    risk_score,
    risk_category,
    created_by_user_id
)
VALUES (
    'rec-maria-001',
    'f47ac10b-58cc-4372-a567-0e02b2c3d479',
    '2024-09-15 10:00:00+00',
    '{
        "dietary": {
            "recommendations": [
                "Reduce refined carbohydrate intake to <45% of total calories",
                "Increase fiber intake to 25-30g daily through vegetables and whole grains",
                "Limit sodium intake to <2300mg daily to support blood pressure control"
            ],
            "meal_planning": {
                "carb_counting": true,
                "portion_control": true,
                "timing": "consistent meal times to support glucose control"
            }
        },
        "exercise": {
            "aerobic": "150 minutes moderate-intensity exercise per week",
            "resistance": "2-3 sessions per week targeting major muscle groups",
            "monitoring": "Check blood glucose before and after exercise"
        },
        "medication_safety": {
            "current_regimen": "Continue Metformin 500mg BID",
            "monitoring": "Consider medication adjustment if HbA1c remains >8.0%",
            "adherence": "Review medication timing and food interactions"
        },
        "monitoring": {
            "glucose": "Daily fasting glucose, target <130 mg/dL",
            "blood_pressure": "Weekly home BP monitoring, target <130/80",
            "weight": "Daily weight tracking",
            "lab_followup": "Repeat HbA1c in 3 months, target <7.0%"
        },
        "education": {
            "diabetes_self_management": "Enroll in diabetes education program",
            "hypoglycemia_recognition": "Review signs and treatment of low blood sugar"
        }
    }',
    '{
        "risk_interpretation": "Elevated 5-year cardiovascular risk (28%) based on current glucose control and blood pressure patterns",
        "key_contributors": [
            {"factor": "HbA1c 8.1%", "impact": "High", "target": "<7.0%"},
            {"factor": "Blood pressure 138/88", "impact": "Moderate", "target": "<130/80"},
            {"factor": "LDL cholesterol 118 mg/dL", "impact": "Moderate", "target": "<100 mg/dL"},
            {"factor": "BMI 29.9", "impact": "Moderate", "target": "5-10% weight loss"}
        ],
        "model_confidence": 0.82,
        "evidence_strength": "High - based on ADA guidelines and multiple RCTs"
    }',
    'risk_lgbm_v0.1',
    'pending',
    0.28,
    'high',
    'c47ac10b-58cc-4372-a567-0e02b2c3d480'
);

-- Evidence links for Maria's recommendation
INSERT INTO evidence_links (recommendation_id, source_type, url, title, weight, snippet, metadata)
VALUES 
    (
        'rec-maria-001',
        'guideline',
        'https://diabetesjournals.org/care/article/47/Supplement_1/S1/153277/Introduction-and-Methodology-Standards-of-Care-in',
        'ADA Standards of Care in Diabetes 2024',
        0.95,
        'HbA1c target <7.0% for most adults with diabetes to reduce microvascular complications',
        '{"guideline_version": "2024", "recommendation_class": "Class I", "evidence_level": "A"}'
    ),
    (
        'rec-maria-001',
        'pubmed',
        'https://pubmed.ncbi.nlm.nih.gov/32640374/',
        'Cardiovascular Risk Reduction with Intensive Lifestyle Intervention in Type 2 Diabetes',
        0.85,
        'Intensive lifestyle intervention reduced major cardiovascular events by 21% in adults with type 2 diabetes',
        '{"pmid": "32640374", "study_type": "RCT", "n_participants": 5145, "followup_years": 9.6}'
    ),
    (
        'rec-maria-001',
        'pubmed',
        'https://pubmed.ncbi.nlm.nih.gov/33667375/',
        'Blood Pressure Targets in Diabetes: A Clinical Practice Guideline Update',
        0.80,
        'Target blood pressure <130/80 mmHg recommended for adults with diabetes and established cardiovascular disease risk',
        '{"pmid": "33667375", "study_type": "guideline", "organization": "ADA/ESC"}'
    ),
    (
        'rec-maria-001',
        'rxnorm',
        'https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/020357s037,021202s021lbl.pdf',
        'Metformin Hydrochloride - FDA Label',
        0.75,
        'Metformin is indicated as adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes',
        '{"rxcui": "6809", "drug_class": "biguanide", "safety_profile": "well_established"}'
    );

-- Audit log entries for Maria's case
SELECT create_audit_log(
    'system',
    NULL,
    'recommendation.created',
    'recommendation',
    'rec-maria-001'::uuid,
    '{"risk_score": 0.28, "model_version": "risk_lgbm_v0.1", "evidence_score": 0.84}'::jsonb,
    'risk_lgbm_v0.1'
);

-- Additional synthetic patients for testing

-- Low-risk patient: John Smith
INSERT INTO patients (id, user_id, demographics, consent)
VALUES (
    'a47ac10b-58cc-4372-a567-0e02b2c3d481',
    'a47ac10b-58cc-4372-a567-0e02b2c3d481',
    '{
        "name": "John Smith",
        "age": 42,
        "sex": "M",
        "ethnicity": "Caucasian",
        "height_cm": 180,
        "weight_kg": 75.0,
        "bmi": 23.1,
        "family_history": {
            "diabetes": false,
            "heart_disease": false,
            "hypertension": false
        },
        "lifestyle": {
            "smoking": "never",
            "alcohol": "moderate",
            "exercise_mins_week": 240
        }
    }',
    '{"ehr_access": true, "data_sharing": true, "research_participation": true}'
);

INSERT INTO user_roles (user_id, role, metadata)
VALUES ('a47ac10b-58cc-4372-a567-0e02b2c3d481', 'patient', '{}');

INSERT INTO clinician_patients (clinician_user_id, patient_id)
VALUES ('c47ac10b-58cc-4372-a567-0e02b2c3d480', 'a47ac10b-58cc-4372-a567-0e02b2c3d481');

-- John's vitals (healthy ranges)
INSERT INTO vitals (patient_id, ts, type, value, unit, source)
VALUES 
    ('a47ac10b-58cc-4372-a567-0e02b2c3d481', '2024-09-01 08:00:00+00', 'glucose_fasting', 92, 'mg/dL', 'device'),
    ('a47ac10b-58cc-4372-a567-0e02b2c3d481', '2024-09-01 09:00:00+00', 'sbp', 118, 'mmHg', 'device'),
    ('a47ac10b-58cc-4372-a567-0e02b2c3d481', '2024-09-01 09:00:00+00', 'dbp', 75, 'mmHg', 'device'),
    ('a47ac10b-58cc-4372-a567-0e02b2c3d481', '2024-09-01 07:30:00+00', 'weight', 75.0, 'kg', 'device'),
    ('a47ac10b-58cc-4372-a567-0e02b2c3d481', '2024-09-01 10:00:00+00', 'total_cholesterol', 165, 'mg/dL', 'lab'),
    ('a47ac10b-58cc-4372-a567-0e02b2c3d481', '2024-09-01 10:00:00+00', 'ldl_cholesterol', 95, 'mg/dL', 'lab'),
    ('a47ac10b-58cc-4372-a567-0e02b2c3d481', '2024-09-01 10:00:00+00', 'hdl_cholesterol', 55, 'mg/dL', 'lab');

-- Moderate-risk patient: Patricia Lee (pre-diabetes)
INSERT INTO patients (id, user_id, demographics, consent)
VALUES (
    'b47ac10b-58cc-4372-a567-0e02b2c3d482',
    'b47ac10b-58cc-4372-a567-0e02b2c3d482',
    '{
        "name": "Patricia Lee",
        "age": 51,
        "sex": "F",
        "ethnicity": "Asian",
        "height_cm": 158,
        "weight_kg": 68.0,
        "bmi": 27.2,
        "family_history": {
            "diabetes": true,
            "heart_disease": false,
            "hypertension": true
        },
        "lifestyle": {
            "smoking": "former",
            "alcohol": "rare",
            "exercise_mins_week": 90
        }
    }',
    '{"ehr_access": true, "data_sharing": false, "research_participation": false}'
);

INSERT INTO user_roles (user_id, role, metadata)
VALUES ('b47ac10b-58cc-4372-a567-0e02b2c3d482', 'patient', '{}');

INSERT INTO clinician_patients (clinician_user_id, patient_id)
VALUES ('c47ac10b-58cc-4372-a567-0e02b2c3d480', 'b47ac10b-58cc-4372-a567-0e02b2c3d482');

-- Patricia's conditions (pre-diabetes)
INSERT INTO conditions (patient_id, icd10_code, name, severity, onset_date, active)
VALUES 
    ('b47ac10b-58cc-4372-a567-0e02b2c3d482', 'R73.03', 'Prediabetes', 'mild', '2024-06-15', true);

-- Patricia's vitals (borderline values)
INSERT INTO vitals (patient_id, ts, type, value, unit, source)
VALUES 
    ('b47ac10b-58cc-4372-a567-0e02b2c3d482', '2024-09-01 08:00:00+00', 'glucose_fasting', 118, 'mg/dL', 'device'),
    ('b47ac10b-58cc-4372-a567-0e02b2c3d482', '2024-09-01 10:00:00+00', 'hba1c', 6.2, '%', 'lab'),
    ('b47ac10b-58cc-4372-a567-0e02b2c3d482', '2024-09-01 09:00:00+00', 'sbp', 128, 'mmHg', 'device'),
    ('b47ac10b-58cc-4372-a567-0e02b2c3d482', '2024-09-01 09:00:00+00', 'dbp', 82, 'mmHg', 'device'),
    ('b47ac10b-58cc-4372-a567-0e02b2c3d482', '2024-09-01 07:30:00+00', 'weight', 68.0, 'kg', 'device');
