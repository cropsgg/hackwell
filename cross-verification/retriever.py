"""Hybrid retrieval system combining semantic and lexical search."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import structlog
import asyncpg
from pydantic import BaseModel

logger = structlog.get_logger()


class RetrievalResult(BaseModel):
    """Result of retrieval operation."""
    chunk_id: str
    document_id: str
    content: str
    source_type: str
    title: Optional[str] = None
    url: Optional[str] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None
    pub_date: Optional[str] = None
    semantic_score: float
    lexical_score: float
    combined_score: float
    metadata: Dict[str, Any] = {}


class HybridRetriever:
    """Hybrid retrieval system using semantic + lexical search with RRF."""
    
    def __init__(
        self,
        db_pool: asyncpg.Pool,
        embeddings_provider,
        k_semantic: int = 8,
        k_lexical: int = 8,
        rrf_k: int = 60
    ):
        self.db_pool = db_pool
        self.embeddings_provider = embeddings_provider
        self.k_semantic = k_semantic
        self.k_lexical = k_lexical
        self.rrf_k = rrf_k
    
    async def retrieve(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 16
    ) -> List[RetrievalResult]:
        """Retrieve relevant passages using hybrid search."""
        try:
            # Generate query embedding
            embedding_result = await self.embeddings_provider.embed_text(query)
            query_embedding = embedding_result.embedding
            
            # Perform hybrid search
            async with self.db_pool.acquire() as conn:
                # Use the hybrid_search function from schema
                results = await conn.fetch(
                    """
                    SELECT 
                        chunk_id,
                        document_id,
                        content,
                        semantic_score,
                        lexical_score,
                        combined_score
                    FROM hybrid_search($1, $2, $3, $4)
                    """,
                    query,
                    query_embedding,
                    self.k_semantic,
                    self.k_lexical
                )
                
                # Get additional document metadata
                if results:
                    chunk_ids = [str(row['chunk_id']) for row in results]
                    metadata_query = """
                    SELECT 
                        ec.id as chunk_id,
                        ec.document_id,
                        ed.source_type,
                        ed.title,
                        ed.url,
                        ed.pmid,
                        ed.doi,
                        ed.pub_date,
                        ed.quality
                    FROM evidence_chunks ec
                    JOIN evidence_documents ed ON ec.document_id = ed.id
                    WHERE ec.id = ANY($1)
                    """
                    metadata_results = await conn.fetch(metadata_query, chunk_ids)
                    
                    # Create metadata lookup
                    metadata_lookup = {
                        str(row['chunk_id']): {
                            'source_type': row['source_type'],
                            'title': row['title'],
                            'url': row['url'],
                            'pmid': row['pmid'],
                            'doi': row['doi'],
                            'pub_date': row['pub_date'].isoformat() if row['pub_date'] else None,
                            'quality': row['quality']
                        }
                        for row in metadata_results
                    }
                else:
                    metadata_lookup = {}
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for row in results:
                chunk_id = str(row['chunk_id'])
                metadata = metadata_lookup.get(chunk_id, {})
                
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=str(row['document_id']),
                    content=row['content'],
                    source_type=metadata.get('source_type', 'unknown'),
                    title=metadata.get('title'),
                    url=metadata.get('url'),
                    pmid=metadata.get('pmid'),
                    doi=metadata.get('doi'),
                    pub_date=metadata.get('pub_date'),
                    semantic_score=float(row['semantic_score']),
                    lexical_score=float(row['lexical_score']),
                    combined_score=float(row['combined_score']),
                    metadata={
                        'quality': metadata.get('quality'),
                        'query': query
                    }
                )
                retrieval_results.append(result)
            
            # Apply filters if provided
            if filters:
                retrieval_results = self._apply_filters(retrieval_results, filters)
            
            # Limit results
            retrieval_results = retrieval_results[:max_results]
            
            logger.info("Hybrid retrieval completed",
                       query=query[:50] + "...",
                       results_count=len(retrieval_results),
                       semantic_k=self.k_semantic,
                       lexical_k=self.k_lexical)
            
            return retrieval_results
            
        except Exception as e:
            logger.error("Hybrid retrieval failed", query=query, error=str(e))
            return []
    
    def _apply_filters(self, results: List[RetrievalResult], filters: Dict[str, Any]) -> List[RetrievalResult]:
        """Apply filters to retrieval results."""
        filtered_results = results
        
        # Filter by source type
        if 'source_types' in filters:
            allowed_types = filters['source_types']
            filtered_results = [
                r for r in filtered_results 
                if r.source_type in allowed_types
            ]
        
        # Filter by date range
        if 'min_date' in filters or 'max_date' in filters:
            filtered_results = [
                r for r in filtered_results
                if self._date_in_range(r.pub_date, filters.get('min_date'), filters.get('max_date'))
            ]
        
        # Filter by quality score
        if 'min_quality' in filters:
            filtered_results = [
                r for r in filtered_results
                if r.metadata.get('quality', 0) >= filters['min_quality']
            ]
        
        return filtered_results
    
    def _date_in_range(self, date_str: Optional[str], min_date: Optional[str], max_date: Optional[str]) -> bool:
        """Check if date is within range."""
        if not date_str:
            return True
        
        try:
            from datetime import datetime
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            if min_date:
                min_dt = datetime.fromisoformat(min_date.replace('Z', '+00:00'))
                if date < min_dt:
                    return False
            
            if max_date:
                max_dt = datetime.fromisoformat(max_date.replace('Z', '+00:00'))
                if date > max_dt:
                    return False
            
            return True
        except:
            return True
    
    async def retrieve_by_claim(
        self, 
        claim: str, 
        context: Dict[str, Any],
        max_results: int = 16
    ) -> List[RetrievalResult]:
        """Retrieve evidence for a specific claim with context."""
        # Build enhanced query with context
        enhanced_query = self._build_enhanced_query(claim, context)
        
        # Set filters based on context
        filters = self._build_context_filters(context)
        
        return await self.retrieve(enhanced_query, filters, max_results)
    
    def _build_enhanced_query(self, claim: str, context: Dict[str, Any]) -> str:
        """Build enhanced query with context information."""
        query_parts = [claim]
        
        # Add context information
        if context.get('conditions'):
            query_parts.extend(context['conditions'])
        
        if context.get('age'):
            age = context['age']
            if age < 18:
                query_parts.append("pediatric")
            elif age > 65:
                query_parts.append("elderly")
        
        if context.get('ckd'):
            query_parts.append("chronic kidney disease")
        
        if context.get('pregnancy'):
            query_parts.append("pregnancy")
        
        return " ".join(query_parts)
    
    def _build_context_filters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build filters based on patient context."""
        filters = {}
        
        # Prefer recent evidence
        from datetime import datetime, timedelta
        filters['min_date'] = (datetime.now() - timedelta(days=3650)).isoformat()  # 10 years
        
        # Prefer higher quality sources
        filters['min_quality'] = 0.3
        
        return filters


class MockRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, **kwargs):
        self.db_pool = None
        self.embeddings_provider = None
        self.k_semantic = kwargs.get('k_semantic', 8)
        self.k_lexical = kwargs.get('k_lexical', 8)
    
    async def retrieve(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 16
    ) -> List[RetrievalResult]:
        """Mock retrieval returning sample results."""
        # Generate mock results based on query
        mock_results = []
        
        # Simple keyword-based mock results
        if "exercise" in query.lower():
            mock_results.append(RetrievalResult(
                chunk_id="mock_1",
                document_id="doc_1",
                content="Regular aerobic exercise improves glycemic control in type 2 diabetes patients.",
                source_type="rct",
                title="Exercise and Diabetes Management",
                url="https://example.com/exercise-study",
                semantic_score=0.85,
                lexical_score=0.75,
                combined_score=0.80,
                metadata={"quality": 0.8}
            ))
        
        if "metformin" in query.lower():
            mock_results.append(RetrievalResult(
                chunk_id="mock_2",
                document_id="doc_2",
                content="Metformin is the first-line treatment for type 2 diabetes with proven efficacy and safety.",
                source_type="guideline",
                title="ADA Standards of Care",
                url="https://example.com/ada-guidelines",
                semantic_score=0.90,
                lexical_score=0.80,
                combined_score=0.85,
                metadata={"quality": 0.9}
            ))
        
        if "sulfonylurea" in query.lower():
            mock_results.append(RetrievalResult(
                chunk_id="mock_3",
                document_id="doc_3",
                content="Sulfonylureas may cause hypoglycemia and should be used with caution in elderly patients.",
                source_type="label_warning",
                title="Drug Safety Information",
                url="https://example.com/drug-warning",
                semantic_score=0.70,
                lexical_score=0.65,
                combined_score=0.68,
                metadata={"quality": 0.7}
            ))
        
        # Apply filters if provided
        if filters:
            mock_results = self._apply_filters(mock_results, filters)
        
        return mock_results[:max_results]
    
    def _apply_filters(self, results: List[RetrievalResult], filters: Dict[str, Any]) -> List[RetrievalResult]:
        """Apply filters to mock results."""
        filtered_results = results
        
        if 'source_types' in filters:
            allowed_types = filters['source_types']
            filtered_results = [
                r for r in filtered_results 
                if r.source_type in allowed_types
            ]
        
        return filtered_results
    
    async def retrieve_by_claim(
        self, 
        claim: str, 
        context: Dict[str, Any],
        max_results: int = 16
    ) -> List[RetrievalResult]:
        """Mock retrieval by claim."""
        return await self.retrieve(claim, None, max_results)


def create_retriever(
    retriever_type: str = "hybrid",
    db_pool: Optional[asyncpg.Pool] = None,
    embeddings_provider=None,
    **kwargs
) -> Any:
    """Factory function to create retriever."""
    
    if retriever_type == "hybrid":
        if not db_pool or not embeddings_provider:
            raise ValueError("Database pool and embeddings provider required for hybrid retriever")
        return HybridRetriever(db_pool, embeddings_provider, **kwargs)
    
    elif retriever_type == "mock":
        return MockRetriever(**kwargs)
    
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
