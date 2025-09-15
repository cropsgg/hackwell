# Evidence Verifier (RAG) — Cross-Verification Service

A comprehensive evidence verification system that validates care plan recommendations using RAG (Retrieval-Augmented Generation) with multiple evidence sources including ADA Standards of Care, PubMed, openFDA, and RxNorm.

## Overview

This service implements the PRD's cross-verification pipeline that every recommendation must pass. It provides:

- **Claim Extraction**: Converts care plan recommendations into verifiable claims
- **Hybrid Retrieval**: Combines semantic (vector) and lexical (full-text) search
- **Stance Classification**: Uses NLI to determine if evidence supports/contradicts claims
- **Confidence Scoring**: Aggregates evidence with weighted scoring
- **External API Integration**: PubMed, openFDA, RxNorm, and ADA guidelines
- **Audit Trail**: Complete verification history and evidence links

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Care Plan     │───▶│  Claim Extractor │───▶│   Evidence      │
│  Recommendations│    │                  │    │   Retrieval     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Verification  │◀───│  Stance Classifier│◀───│  Vector DB +    │
│     Result      │    │                  │    │  External APIs  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Features

### Evidence Sources
- **ADA Standards of Care**: Curated guideline statements with citations
- **PubMed**: Scientific literature via E-utilities API
- **openFDA**: Drug safety information and labeling
- **RxNorm**: Drug normalization and interaction checking
- **Vector Database**: Curated corpus with semantic search

### Verification Pipeline
1. **Claim Extraction**: Rule-based extraction of verifiable claims
2. **Hybrid Retrieval**: Semantic + lexical search with RRF merging
3. **Stance Classification**: NLI-based support/contradict/neutral classification
4. **Confidence Scoring**: Weighted aggregation with source quality factors
5. **Verdict Determination**: Approved/flagged based on evidence thresholds

### Scoring System
- **Source Weights**: Guidelines (0.9), RCTs (0.8), Cohort (0.6), Case (0.3)
- **Quality Factors**: Recency, sample size, journal impact, peer review
- **Verdict Thresholds**: Contradict ≥0.55 → FLAG, Support ≥0.60 → APPROVE

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key (for embeddings)

### Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Database setup**:
```sql
-- Run the schema.sql file
\i schema.sql
```

3. **Environment variables**:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/hackwell"
export OPENAI_API_KEY="your-openai-api-key"
export PUBMED_EMAIL="your-email@example.com"
```

4. **Populate evidence database**:
```bash
python ingest.py
```

## Usage

### FastAPI Service

Start the service:
```bash
python app.py
```

### API Endpoints

#### Verify Evidence
```http
POST /verify
Content-Type: application/json

{
  "recommendation_id": "rec-123",
  "care_plan": {
    "exercise": {
      "aerobic": "30 minutes moderate activity 5 days/week"
    },
    "medication_safety": {
      "warnings": ["Avoid sulfonylureas due to hypoglycemia risk"]
    }
  },
  "patient_context": {
    "age": 55,
    "conditions": ["type 2 diabetes"],
    "medications": ["metformin"]
  }
}
```

#### Get Evidence
```http
GET /evidence/{recommendation_id}
```

#### Health Check
```http
GET /health
```

### Python API

```python
from cross_verification.service import create_verification_service
from cross_verification.models import VerificationRequest

# Create service
service = create_verification_service(db_pool, config)

# Verify recommendation
request = VerificationRequest(
    recommendation_id="rec-123",
    care_plan=care_plan,
    patient_context=patient_context
)

result = await service.verify_recommendation(request)
print(f"Status: {result['overall_status']}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Required |
| `EMBEDDINGS_MODEL` | Embedding model | `text-embedding-3-large` |
| `STANCE_MODEL` | NLI model for stance classification | `microsoft/deberta-base-mnli` |
| `K_SEMANTIC` | Semantic retrieval count | `8` |
| `K_LEXICAL` | Lexical retrieval count | `8` |
| `ENABLE_PUBMED` | Enable PubMed integration | `true` |
| `ENABLE_OPENFDA` | Enable openFDA integration | `true` |
| `ENABLE_RXNORM` | Enable RxNorm integration | `true` |
| `ENABLE_ADA` | Enable ADA guidelines | `true` |

### Scoring Configuration

```python
# Source weights
source_weights = {
    "guideline": 0.9,
    "rct": 0.8,
    "cohort": 0.6,
    "case": 0.3,
    "label_warning": 1.0
}

# Verdict thresholds
contradict_threshold = 0.55
support_threshold = 0.60
contradict_safety_threshold = 0.40
```

## Testing

Run the test suite:
```bash
python test_verification.py
```

Run individual component tests:
```python
# Test claim extraction
from cross_verification.claim_extractor import create_claim_extractor
extractor = create_claim_extractor()
claims = extractor.extract_claims(care_plan, patient_context)

# Test stance classification
from cross_verification.stance import create_stance_classifier
classifier = create_stance_classifier()
result = await classifier.classify(claim, passage)

# Test evidence scoring
from cross_verification.scorer import create_evidence_scorer
scorer = create_evidence_scorer()
score = scorer.score_individual_evidence(...)
```

## Database Schema

### Core Tables

- `evidence_documents`: Source documents with metadata
- `evidence_chunks`: Chunked content with vector embeddings
- `evidence_links`: Links between recommendations and evidence
- `audit_logs`: Verification events and audit trail

### Vector Search

Uses pgvector for semantic similarity search with cosine distance:
```sql
CREATE INDEX ON evidence_chunks USING ivfflat (embedding vector_cosine_ops);
```

### Hybrid Search Function

```sql
SELECT * FROM hybrid_search(
    'diabetes exercise benefits',
    query_embedding,
    k_semantic => 8,
    k_lexical => 8
);
```

## Performance

### Latency Targets
- p95 < 3-5 seconds per recommendation
- k=16 passages per claim
- Cached external API results

### Optimization
- Vector index with IVFFlat (lists=100)
- Full-text search with GIN index
- Batch processing for external APIs
- Connection pooling for database

## Monitoring

### Health Checks
- Database connectivity
- External API availability
- Model loading status
- Service metrics

### Audit Trail
- Complete verification history
- Evidence source tracking
- Model version tracking
- Performance metrics

## Development

### Project Structure
```
cross-verification/
├── adapters/          # External API clients
├── embeddings.py      # Vector embeddings
├── stance.py         # Stance classification
├── claim_extractor.py # Claim extraction
├── retriever.py      # Hybrid retrieval
├── scorer.py         # Evidence scoring
├── service.py        # Main orchestrator
├── app.py           # FastAPI application
├── models.py        # Pydantic models
├── config.py        # Configuration
├── ingest.py        # Data ingestion
└── test_verification.py # Tests
```

### Adding New Evidence Sources

1. Create adapter in `adapters/`
2. Add to service configuration
3. Update ingestion script
4. Add tests

### Customizing Scoring

1. Modify weights in `scorer.py`
2. Update verdict thresholds
3. Add new quality factors
4. Test with sample data

## License

This project is part of the Hackwell AI Wellness Assistant system.

## Support

For issues and questions:
1. Check the test suite for examples
2. Review the configuration options
3. Check the audit logs for debugging
4. Contact the development team
