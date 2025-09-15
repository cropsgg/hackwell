"""Embeddings provider for vector similarity search."""

import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import structlog
import openai
from pydantic import BaseModel

logger = structlog.get_logger()


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""
    embedding: List[float]
    model: str
    usage_tokens: int
    cached: bool = False


class EmbeddingsProvider(ABC):
    """Abstract base class for embeddings providers."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass


class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    """OpenAI text-embedding-3-large provider."""
    
    def __init__(
        self, 
        api_key: str,
        model: str = "text-embedding-3-large",
        max_retries: int = 3,
        cache_embeddings: bool = True
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.cache_embeddings = cache_embeddings
        self._cache = {} if cache_embeddings else None
        
        # Model dimensions
        self._dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
    
    def get_dimension(self) -> int:
        """Get embedding dimension for this model."""
        return self._dimensions.get(self.model, 1536)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        if self.cache_embeddings and self._cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                logger.debug("Using cached embedding", cache_key=cache_key[:8])
                return EmbeddingResult(
                    embedding=self._cache[cache_key]["embedding"],
                    model=self.model,
                    usage_tokens=self._cache[cache_key]["usage_tokens"],
                    cached=True
                )
        
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            usage_tokens = response.usage.total_tokens
            
            # Cache the result
            if self.cache_embeddings and self._cache is not None:
                cache_key = self._get_cache_key(text)
                self._cache[cache_key] = {
                    "embedding": embedding,
                    "usage_tokens": usage_tokens
                }
            
            logger.debug("Generated embedding", 
                        model=self.model, 
                        tokens=usage_tokens,
                        dimension=len(embedding))
            
            return EmbeddingResult(
                embedding=embedding,
                model=self.model,
                usage_tokens=usage_tokens,
                cached=False
            )
            
        except Exception as e:
            logger.error("Failed to generate embedding", 
                        model=self.model, 
                        error=str(e))
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        # Check cache first
        cached_results = []
        texts_to_embed = []
        cache_indices = []
        
        if self.cache_embeddings and self._cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    cached_results.append(EmbeddingResult(
                        embedding=self._cache[cache_key]["embedding"],
                        model=self.model,
                        usage_tokens=self._cache[cache_key]["usage_tokens"],
                        cached=True
                    ))
                    cache_indices.append(i)
                else:
                    texts_to_embed.append(text)
        else:
            texts_to_embed = texts
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts_to_embed,
                    encoding_format="float"
                )
                
                # Process results
                new_results = []
                for i, data in enumerate(response.data):
                    embedding = data.embedding
                    usage_tokens = response.usage.total_tokens // len(texts_to_embed)
                    
                    # Cache the result
                    if self.cache_embeddings and self._cache is not None:
                        cache_key = self._get_cache_key(texts_to_embed[i])
                        self._cache[cache_key] = {
                            "embedding": embedding,
                            "usage_tokens": usage_tokens
                        }
                    
                    new_results.append(EmbeddingResult(
                        embedding=embedding,
                        model=self.model,
                        usage_tokens=usage_tokens,
                        cached=False
                    ))
                
                logger.debug("Generated batch embeddings", 
                            model=self.model, 
                            total_texts=len(texts),
                            cached=len(cached_results),
                            new=len(new_results))
                
            except Exception as e:
                logger.error("Failed to generate batch embeddings", 
                            model=self.model, 
                            error=str(e))
                raise
        
        # Combine cached and new results in correct order
        all_results = [None] * len(texts)
        cached_idx = 0
        new_idx = 0
        
        for i in range(len(texts)):
            if i in cache_indices:
                all_results[i] = cached_results[cached_idx]
                cached_idx += 1
            else:
                all_results[i] = new_results[new_idx]
                new_idx += 1
        
        return all_results


class MockEmbeddingsProvider(EmbeddingsProvider):
    """Mock embeddings provider for testing."""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
    
    def get_dimension(self) -> int:
        return self.dimension
    
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate mock embedding."""
        # Simple hash-based mock embedding
        hash_obj = hashlib.sha256(text.encode())
        embedding = [
            (int(hash_obj.hexdigest()[i:i+2], 16) / 255.0 - 0.5) * 2
            for i in range(0, min(len(hash_obj.hexdigest()), self.dimension * 2), 2)
        ]
        
        # Pad or truncate to correct dimension
        if len(embedding) < self.dimension:
            embedding.extend([0.0] * (self.dimension - len(embedding)))
        else:
            embedding = embedding[:self.dimension]
        
        return EmbeddingResult(
            embedding=embedding,
            model="mock",
            usage_tokens=len(text.split()),
            cached=False
        )
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate mock embeddings for multiple texts."""
        return [await self.embed_text(text) for text in texts]


def create_embeddings_provider(
    provider_type: str = "openai",
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-large",
    **kwargs
) -> EmbeddingsProvider:
    """Factory function to create embeddings provider."""
    
    if provider_type == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAIEmbeddingsProvider(api_key=api_key, model=model, **kwargs)
    
    elif provider_type == "mock":
        dimension = kwargs.get("dimension", 1536)
        return MockEmbeddingsProvider(dimension=dimension)
    
    else:
        raise ValueError(f"Unknown embeddings provider: {provider_type}")
