-- Evidence Verifier (RAG) Database Schema
-- Supabase/PostgreSQL with pgvector extension

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
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
    embedding vector(1536),           -- OpenAI text-embedding-3-large dimension
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

-- Vector similarity search index
CREATE INDEX ON evidence_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Hash index for deduplication
CREATE INDEX ON evidence_chunks(hash);

-- Update evidence_links table to match PRD requirements
ALTER TABLE evidence_links 
ADD COLUMN IF NOT EXISTS stance text,                      -- support|contradict|neutral|warning
ADD COLUMN IF NOT EXISTS score numeric,                    -- combined confidence 0..1
ADD COLUMN IF NOT EXISTS snippet text,
ADD COLUMN IF NOT EXISTS pmid text,
ADD COLUMN IF NOT EXISTS doi text,
ADD COLUMN IF NOT EXISTS added_at timestamptz DEFAULT now();

-- Add constraint for stance values
ALTER TABLE evidence_links 
ADD CONSTRAINT IF NOT EXISTS valid_stance CHECK (stance IN ('support', 'contradict', 'neutral', 'warning'));

-- Update audit_logs table for evidence verification events
ALTER TABLE audit_logs 
ADD COLUMN IF NOT EXISTS model_version text,
ADD COLUMN IF NOT EXISTS ts timestamptz DEFAULT now();

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

-- Create function for hybrid search (semantic + lexical)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text text,
    query_embedding vector(1536),
    k_semantic int DEFAULT 8,
    k_lexical int DEFAULT 8
)
RETURNS TABLE (
    chunk_id uuid,
    document_id uuid,
    content text,
    semantic_score float,
    lexical_score float,
    combined_score float
) AS $$
BEGIN
    RETURN QUERY
    WITH semantic_results AS (
        SELECT 
            ec.id as chunk_id,
            ec.document_id,
            ec.content,
            1 - (ec.embedding <=> query_embedding) as semantic_score,
            0.0 as lexical_score
        FROM evidence_chunks ec
        ORDER BY ec.embedding <=> query_embedding
        LIMIT k_semantic
    ),
    lexical_results AS (
        SELECT 
            ec.id as chunk_id,
            ec.document_id,
            ec.content,
            0.0 as semantic_score,
            ts_rank(ec.tsv, websearch_to_tsquery('english', query_text)) as lexical_score
        FROM evidence_chunks ec
        WHERE ec.tsv @@ websearch_to_tsquery('english', query_text)
        ORDER BY lexical_score DESC
        LIMIT k_lexical
    ),
    combined_results AS (
        SELECT * FROM semantic_results
        UNION ALL
        SELECT * FROM lexical_results
    ),
    aggregated AS (
        SELECT 
            chunk_id,
            document_id,
            content,
            MAX(semantic_score) as semantic_score,
            MAX(lexical_score) as lexical_score,
            -- Reciprocal Rank Fusion
            (1.0 / (1.0 + MIN(
                CASE WHEN semantic_score > 0 THEN 
                    ROW_NUMBER() OVER (ORDER BY semantic_score DESC)
                ELSE NULL END
            ))) + 
            (1.0 / (1.0 + MIN(
                CASE WHEN lexical_score > 0 THEN 
                    ROW_NUMBER() OVER (ORDER BY lexical_score DESC)
                ELSE NULL END
            ))) as combined_score
        FROM combined_results
        GROUP BY chunk_id, document_id, content
    )
    SELECT 
        aggregated.chunk_id,
        aggregated.document_id,
        aggregated.content,
        aggregated.semantic_score,
        aggregated.lexical_score,
        aggregated.combined_score
    FROM aggregated
    ORDER BY aggregated.combined_score DESC;
END;
$$ LANGUAGE plpgsql;

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
FROM recommendations r
LEFT JOIN evidence_links el ON r.id = el.recommendation_id
GROUP BY r.id, r.patient_id, r.status;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON evidence_documents TO authenticated;
GRANT ALL ON evidence_chunks TO authenticated;
GRANT ALL ON evidence_links TO authenticated;
GRANT EXECUTE ON FUNCTION hybrid_search TO authenticated;
