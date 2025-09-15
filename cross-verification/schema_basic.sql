-- Basic Evidence Verification Database Schema
-- PostgreSQL with basic setup (without pgvector for now)

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Evidence documents and chunks for RAG
CREATE TABLE evidence_documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type text NOT NULL,        -- guideline|rct|cohort|case|label_warning
    title text,
    url text,
    doi text,
    pmid text,
    pub_date date,
    jurisdiction text,
    license text,
    section text,
    quality numeric,                  -- optional curator score 0..1
    created_at timestamptz DEFAULT now(),
    
    CONSTRAINT valid_source_type CHECK (source_type IN ('guideline', 'rct', 'cohort', 'case', 'label_warning'))
);

CREATE TABLE evidence_chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES evidence_documents(id) ON DELETE CASCADE,
    chunk_index int,
    content text NOT NULL,
    tsv tsvector,                     -- for lexical retrieval
    embedding text,                   -- store as text for now (will be vector later)
    embedding_version text DEFAULT 'v1',
    hash text UNIQUE,
    created_at timestamptz DEFAULT now()
);

-- Indexes for performance
CREATE INDEX ON evidence_documents(source_type);
CREATE INDEX ON evidence_documents(pub_date);
CREATE INDEX ON evidence_documents(quality);

-- Full-text search index
CREATE INDEX ON evidence_chunks USING gin(tsv);

-- Hash index for deduplication
CREATE INDEX ON evidence_chunks(hash);

-- Update evidence_links table to match PRD requirements
CREATE TABLE IF NOT EXISTS evidence_links (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_id uuid NOT NULL,
    source_type text NOT NULL,
    url text,
    title text,
    weight numeric DEFAULT 0.0,
    snippet text,
    metadata jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    stance text,
    score numeric,
    pmid text,
    doi text,
    added_at timestamptz DEFAULT now()
);

-- Add constraint for stance values
ALTER TABLE evidence_links 
ADD CONSTRAINT valid_stance CHECK (stance IN ('support', 'contradict', 'neutral', 'warning'));

-- Update audit_logs table for evidence verification events
CREATE TABLE IF NOT EXISTS audit_logs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_type text NOT NULL,
    actor_id uuid,
    event text NOT NULL,
    entity_type text,
    entity_id uuid,
    payload jsonb DEFAULT '{}',
    payload_hash text,
    model_version text,
    ts timestamptz DEFAULT now(),
    
    CONSTRAINT valid_actor_type CHECK (actor_type IN ('system', 'clinician', 'patient', 'admin'))
);

-- Create function to update tsvector for evidence chunks
CREATE OR REPLACE FUNCTION update_evidence_chunk_tsv()
RETURNS TRIGGER AS $$
BEGIN
    NEW.tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update tsvector
CREATE TRIGGER update_evidence_chunk_tsv_trigger
    BEFORE INSERT OR UPDATE ON evidence_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_evidence_chunk_tsv();

-- Create view for evidence verification results
CREATE OR REPLACE VIEW evidence_verification_summary AS
SELECT 
    r.id as recommendation_id,
    r.patient_id,
    r.status,
    COUNT(el.id) as total_evidence,
    COUNT(CASE WHEN el.stance = 'support' THEN 1 END) as supporting_evidence,
    COUNT(CASE WHEN el.stance = 'contradict' THEN 1 END) as contradicting_evidence,
    COUNT(CASE WHEN el.stance = 'warning' THEN 1 END) as warning_evidence,
    AVG(el.score) as avg_evidence_score,
    MAX(el.added_at) as last_verification
FROM evidence_links el
LEFT JOIN (SELECT gen_random_uuid() as id, gen_random_uuid() as patient_id, 'pending' as status) r ON true
GROUP BY r.id, r.patient_id, r.status;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO postgres;
GRANT ALL ON evidence_documents TO postgres;
GRANT ALL ON evidence_chunks TO postgres;
GRANT ALL ON evidence_links TO postgres;
