"""Evidence ingestion script for populating the vector database."""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
import asyncpg
from pydantic import BaseModel

from embeddings import create_embeddings_provider
from adapters.pubmed_client import create_pubmed_client
from adapters.openfda_client import create_openfda_client
from adapters.ada_mapper import create_ada_mapper
from config import get_config

logger = structlog.get_logger()


class EvidenceDocument(BaseModel):
    """Evidence document model."""
    source_type: str
    title: str
    url: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pub_date: Optional[str] = None
    jurisdiction: Optional[str] = None
    license: Optional[str] = None
    section: Optional[str] = None
    quality: Optional[float] = None


class EvidenceChunk(BaseModel):
    """Evidence chunk model."""
    content: str
    chunk_index: int
    embedding: List[float]
    hash: str


class EvidenceIngester:
    """Evidence ingestion service."""
    
    def __init__(self, db_pool: asyncpg.Pool, embeddings_provider):
        self.db_pool = db_pool
        self.embeddings_provider = embeddings_provider
        self.chunk_size = 512
        self.chunk_overlap = 0.15
    
    async def ingest_ada_guidelines(self) -> int:
        """Ingest ADA guidelines."""
        try:
            logger.info("Starting ADA guidelines ingestion")
            
            ada_mapper = create_ada_mapper()
            statements = ada_mapper.get_all_statements()
            
            ingested_count = 0
            for statement in statements:
                # Create document
                doc_id = await self._create_document(EvidenceDocument(
                    source_type="guideline",
                    title=statement.title,
                    url=statement.url,
                    doi=None,
                    pmid=None,
                    pub_date=statement.last_updated,
                    jurisdiction="US",
                    license="ADA Standards of Care",
                    section=statement.section,
                    quality=0.9
                ))
                
                if doc_id:
                    # Create chunks
                    chunks = self._create_chunks(statement.content, doc_id)
                    await self._create_chunks_with_embeddings(chunks)
                    ingested_count += 1
            
            logger.info("ADA guidelines ingestion completed", count=ingested_count)
            return ingested_count
            
        except Exception as e:
            logger.error("ADA guidelines ingestion failed", error=str(e))
            return 0
    
    async def ingest_pubmed_articles(
        self, 
        queries: List[str], 
        max_articles_per_query: int = 20
    ) -> int:
        """Ingest PubMed articles."""
        try:
            logger.info("Starting PubMed articles ingestion")
            
            pubmed_client = create_pubmed_client()
            ingested_count = 0
            
            for query in queries:
                articles = await pubmed_client.search_articles(
                    query=query,
                    max_results=max_articles_per_query
                )
                
                for article in articles:
                    # Create document
                    doc_id = await self._create_document(EvidenceDocument(
                        source_type=article.study_type or "observational",
                        title=article.title,
                        url=article.url,
                        doi=article.doi,
                        pmid=article.pmid,
                        pub_date=article.pub_date,
                        jurisdiction="US",
                        license="PubMed",
                        section="abstract",
                        quality=article.quality_score
                    ))
                    
                    if doc_id:
                        # Create chunks from title and abstract
                        content = f"{article.title}. {article.abstract or ''}"
                        chunks = self._create_chunks(content, doc_id)
                        await self._create_chunks_with_embeddings(chunks)
                        ingested_count += 1
            
            logger.info("PubMed articles ingestion completed", count=ingested_count)
            return ingested_count
            
        except Exception as e:
            logger.error("PubMed articles ingestion failed", error=str(e))
            return 0
    
    async def ingest_openfda_labels(self, drug_names: List[str]) -> int:
        """Ingest openFDA drug labels."""
        try:
            logger.info("Starting openFDA labels ingestion")
            
            openfda_client = create_openfda_client()
            ingested_count = 0
            
            for drug_name in drug_names:
                drug_labels = await openfda_client.search_drug(drug_name)
                
                for label in drug_labels:
                    # Create document for warnings
                    if label.warnings:
                        doc_id = await self._create_document(EvidenceDocument(
                            source_type="label_warning",
                            title=f"Safety Information for {drug_name}",
                            url=label.url,
                            doi=None,
                            pmid=None,
                            pub_date=label.last_updated,
                            jurisdiction="US",
                            license="FDA",
                            section="warnings",
                            quality=1.0
                        ))
                        
                        if doc_id:
                            # Create chunks from warnings
                            for warning in label.warnings:
                                chunks = self._create_chunks(warning, doc_id)
                                await self._create_chunks_with_embeddings(chunks)
                                ingested_count += 1
                    
                    # Create document for contraindications
                    if label.contraindications:
                        doc_id = await self._create_document(EvidenceDocument(
                            source_type="label_warning",
                            title=f"Contraindications for {drug_name}",
                            url=label.url,
                            doi=None,
                            pmid=None,
                            pub_date=label.last_updated,
                            jurisdiction="US",
                            license="FDA",
                            section="contraindications",
                            quality=1.0
                        ))
                        
                        if doc_id:
                            # Create chunks from contraindications
                            for contraindication in label.contraindications:
                                chunks = self._create_chunks(contraindication, doc_id)
                                await self._create_chunks_with_embeddings(chunks)
                                ingested_count += 1
            
            logger.info("openFDA labels ingestion completed", count=ingested_count)
            return ingested_count
            
        except Exception as e:
            logger.error("openFDA labels ingestion failed", error=str(e))
            return 0
    
    async def _create_document(self, document: EvidenceDocument) -> Optional[str]:
        """Create a document in the database."""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow(
                    """
                    INSERT INTO evidence_documents (
                        source_type, title, url, doi, pmid, pub_date,
                        jurisdiction, license, section, quality
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                    """,
                    document.source_type,
                    document.title,
                    document.url,
                    document.doi,
                    document.pmid,
                    document.pub_date,
                    document.jurisdiction,
                    document.license,
                    document.section,
                    document.quality
                )
                
                return str(result["id"])
                
        except Exception as e:
            logger.warning("Failed to create document", title=document.title, error=str(e))
            return None
    
    def _create_chunks(self, content: str, document_id: str) -> List[EvidenceChunk]:
        """Create chunks from content."""
        chunks = []
        
        # Simple chunking by character count
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]
            
            # Create hash for deduplication
            chunk_hash = hashlib.md5(f"{document_id}:{chunk_index}:{chunk_content}".encode()).hexdigest()
            
            chunks.append(EvidenceChunk(
                content=chunk_content,
                chunk_index=chunk_index,
                embedding=[],  # Will be filled by create_chunks_with_embeddings
                hash=chunk_hash
            ))
            
            # Move start position with overlap
            start += int(self.chunk_size * (1 - self.chunk_overlap))
            chunk_index += 1
        
        return chunks
    
    async def _create_chunks_with_embeddings(self, chunks: List[EvidenceChunk]):
        """Create chunks with embeddings in the database."""
        try:
            # Generate embeddings for all chunks
            texts = [chunk.content for chunk in chunks]
            embedding_results = await self.embeddings_provider.embed_batch(texts)
            
            # Update chunks with embeddings
            for chunk, embedding_result in zip(chunks, embedding_results):
                chunk.embedding = embedding_result.embedding
            
            # Insert chunks into database
            async with self.db_pool.acquire() as conn:
                for chunk in chunks:
                    await conn.execute(
                        """
                        INSERT INTO evidence_chunks (
                            document_id, chunk_index, content, embedding, hash
                        ) VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (hash) DO NOTHING
                        """,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.embedding,
                        chunk.hash
                    )
            
            logger.debug("Chunks created with embeddings", count=len(chunks))
            
        except Exception as e:
            logger.error("Failed to create chunks with embeddings", error=str(e))
    
    async def run_full_ingestion(self) -> Dict[str, int]:
        """Run full evidence ingestion."""
        logger.info("Starting full evidence ingestion")
        
        results = {
            "ada_guidelines": 0,
            "pubmed_articles": 0,
            "openfda_labels": 0
        }
        
        try:
            # Ingest ADA guidelines
            results["ada_guidelines"] = await self.ingest_ada_guidelines()
            
            # Ingest PubMed articles
            pubmed_queries = [
                "diabetes exercise physical activity",
                "metformin type 2 diabetes",
                "sulfonylurea hypoglycemia",
                "diabetes diet carbohydrate",
                "diabetes monitoring HbA1c",
                "diabetes cardiovascular risk",
                "GLP-1 diabetes treatment",
                "SGLT2 diabetes heart failure"
            ]
            results["pubmed_articles"] = await self.ingest_pubmed_articles(pubmed_queries)
            
            # Ingest openFDA labels
            drug_names = [
                "metformin", "glipizide", "glyburide", "glimepiride",
                "pioglitazone", "rosiglitazone", "sitagliptin",
                "exenatide", "liraglutide", "empagliflozin"
            ]
            results["openfda_labels"] = await self.ingest_openfda_labels(drug_names)
            
            logger.info("Full evidence ingestion completed", results=results)
            
        except Exception as e:
            logger.error("Full evidence ingestion failed", error=str(e))
        
        return results


async def main():
    """Main ingestion function."""
    config = get_config()
    
    # Create database pool
    db_pool = await asyncpg.create_pool(
        config.database_url,
        min_size=5,
        max_size=20
    )
    
    # Create embeddings provider
    embeddings_provider = create_embeddings_provider(
        provider_type=config.embeddings_provider,
        api_key=config.openai_api_key,
        model=config.embeddings_model
    )
    
    # Create ingester
    ingester = EvidenceIngester(db_pool, embeddings_provider)
    
    # Run ingestion
    results = await ingester.run_full_ingestion()
    
    # Close database pool
    await db_pool.close()
    
    print(f"Ingestion completed: {results}")


if __name__ == "__main__":
    asyncio.run(main())
