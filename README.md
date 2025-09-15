# AI Wellness Assistant Backend

A comprehensive backend system for cardiometabolic health risk prediction and personalized care planning.

## Overview

This system provides:
- **Risk Prediction**: ML-powered cardiometabolic risk assessment with SHAP explanations
- **Care Planning**: Evidence-based personalized recommendations
- **Clinical Workflow**: Clinician approval/override with complete audit trails
- **Evidence Verification**: Integration with PubMed, openFDA, RxNorm, and ADA guidelines

## Architecture

```
wellness-backend/
├─ infra/                      # Infrastructure & containers
├─ gateway/                    # FastAPI API Gateway
├─ services/                   # Core microservices
│  ├─ ingest/                  # FHIR/device data ingestion
│  ├─ ml_risk/                 # ML risk prediction models
│  ├─ agents/                  # GenAI agent orchestration
│  ├─ verifier/                # Evidence verification
│  ├─ audit/                   # Audit logging
│  ├─ ws_stream/               # WebSocket streaming
│  └─ jobs/                    # Background tasks
├─ db/                         # Database schema & seeds
└─ scripts/                    # Development utilities
```

## Quick Start

1. **Environment Setup**
   ```bash
   make up            # Start Docker Compose stack
   make db_init       # Initialize database schema
   ```

2. **Generate Synthetic Data**
   ```bash
   python scripts/generate_synthetic.py --n 200
   ```

3. **Train Models**
   ```bash
   make train
   ```

4. **Run Demo**
   ```bash
   make demo
   ```

## Key Features

### Security & Compliance
- Row-level security (RLS) for patient data isolation
- RBAC with clinician-patient assignments
- Complete audit trail for all recommendations
- TLS encryption and secure credential management

### ML Pipeline
- LightGBM/XGBoost for tabular risk prediction
- SHAP explanations for model interpretability
- Calibrated probability outputs
- Comprehensive metrics (AUPRC, ROC, Brier score)

### Evidence-Based Care
- PubMed literature search integration
- FDA adverse event monitoring
- RxNorm drug normalization
- ADA guideline references
- Weighted scoring system for evidence quality

### Clinical Workflow
- Decision support (not autonomous treatment)
- Clinician approval gates for all recommendations
- Override capabilities with justification
- Real-time streaming updates

## Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **Database**: Supabase/PostgreSQL with RLS
- **ML**: LightGBM, XGBoost, SHAP
- **Auth**: Supabase Auth with JWT
- **APIs**: PubMed E-utilities, openFDA, RxNorm
- **Containers**: Docker Compose

## Environment Variables

Required environment variables (see `.env.example`):

```
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
DATABASE_URL=postgresql://...
JWT_AUDIENCE=...
JWT_ISSUER=...
PUBMED_BASE=https://eutils.ncbi.nlm.nih.gov/entrez/eutils
OPENFDA_BASE=https://api.fda.gov
RXNORM_BASE=https://rxnav.nlm.nih.gov/REST
EVIDENCE_MIN_SCORE=0.6
```

## Scope

This implementation covers:
- ✅ Backend services and APIs
- ✅ ML model training and inference
- ✅ Evidence verification pipeline
- ✅ Clinical workflow management
- ✅ Security and compliance features
- ❌ Frontend/UI (separate repository)
- ❌ Autonomous medication dosing

## Development

See individual service READMEs for detailed development instructions.

## License

Proprietary - Hackathon MVP
