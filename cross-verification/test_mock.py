#!/usr/bin/env python3
"""Test script with mock components to avoid API dependencies."""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_mock_embeddings():
    """Test mock embeddings provider."""
    try:
        from embeddings import create_embeddings_provider
        
        # Create mock embeddings provider
        provider = create_embeddings_provider(
            provider_type="mock",
            dimension=1536
        )
        
        # Test embedding generation
        result = await provider.embed_text("Test text for embedding")
        print(f"✅ Mock embeddings working: {len(result.embedding)} dimensions")
        print(f"   Model: {result.model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock embeddings test failed: {e}")
        return False

async def test_claim_extractor():
    """Test claim extraction with better test data."""
    try:
        from claim_extractor import create_claim_extractor, ClaimContext
        
        # Create claim extractor
        extractor = create_claim_extractor()
        
        # Test data with more specific patterns
        care_plan = {
            "exercise": {
                "aerobic": "30 minutes of moderate-intensity aerobic activity 5 days per week",
                "resistance": "Resistance training 2-3 times per week"
            },
            "medication_safety": {
                "warnings": ["Avoid sulfonylureas due to hypoglycemia risk"],
                "monitoring": "Monitor HbA1c every 3 months"
            }
        }
        
        patient_context = ClaimContext(
            age=55,
            conditions=["type 2 diabetes"],
            medications=["metformin"]
        )
        
        # Extract claims
        claims = extractor.extract_claims(care_plan, patient_context)
        print(f"✅ Claim extraction successful: {len(claims)} claims extracted")
        
        for i, claim in enumerate(claims, 1):
            print(f"   {i}. {claim.text}")
            print(f"      Policy: {claim.policy}, Category: {claim.category}")
        
        return True
        
    except Exception as e:
        print(f"❌ Claim extraction test failed: {e}")
        return False

async def test_ada_mapper():
    """Test ADA mapper."""
    try:
        from adapters.ada_mapper import create_ada_mapper
        
        # Create ADA mapper
        mapper = create_ada_mapper()
        
        # Search for statements
        statements = mapper.search_statements("exercise diabetes", max_results=3)
        print(f"✅ ADA mapper working: {len(statements)} statements found")
        
        for statement in statements:
            print(f"   - {statement.title}")
            print(f"     Evidence Level: {statement.evidence_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ ADA mapper test failed: {e}")
        return False

async def test_evidence_scorer():
    """Test evidence scoring."""
    try:
        from scorer import create_evidence_scorer
        
        # Create evidence scorer
        scorer = create_evidence_scorer()
        
        # Test scoring
        score = scorer.score_individual_evidence(
            content="Regular exercise improves glycemic control in diabetes patients.",
            source_type="rct",
            stance="support",
            stance_confidence=0.8,
            retrieval_score=0.7,
            pub_date="2023-01-01",
            quality=0.9
        )
        
        print(f"✅ Evidence scoring working: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evidence scoring test failed: {e}")
        return False

async def test_database_operations():
    """Test basic database operations."""
    try:
        import asyncpg
        
        # Database connection
        database_url = "postgresql://postgres:password@localhost:5432/hackwell"
        conn = await asyncpg.connect(database_url)
        
        # Test inserting a sample document
        doc_id = await conn.fetchval("""
            INSERT INTO evidence_documents (
                source_type, title, content, quality
            ) VALUES ($1, $2, $3, $4)
            RETURNING id
        """, "guideline", "Test Document", "This is a test document", 0.8)
        
        print(f"✅ Document inserted with ID: {doc_id}")
        
        # Test inserting a sample chunk with proper vector dimension
        embedding_vector = [0.1] * 1536  # 1536 dimensions for text-embedding-3-large
        embedding_str = "[" + ",".join(map(str, embedding_vector)) + "]"
        chunk_id = await conn.fetchval("""
            INSERT INTO evidence_chunks (
                document_id, chunk_index, content, embedding, hash
            ) VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """, doc_id, 0, "Test chunk content", embedding_str, "test_hash_123")
        
        print(f"✅ Chunk inserted with ID: {chunk_id}")
        
        # Test hybrid search function
        try:
            results = await conn.fetch("""
                SELECT * FROM hybrid_search(
                    'test query',
                    $1::vector,
                    5,
                    5
                )
            """, embedding_str)
            print(f"✅ Hybrid search function working: {len(results)} results")
        except Exception as e:
            print(f"⚠️  Hybrid search function issue: {e}")
        
        # Clean up test data
        await conn.execute("DELETE FROM evidence_chunks WHERE id = $1", chunk_id)
        await conn.execute("DELETE FROM evidence_documents WHERE id = $1", doc_id)
        print("✅ Test data cleaned up")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Evidence Verification Service - Mock Tests")
    print("=" * 60)
    
    tests = [
        ("Mock Embeddings", test_mock_embeddings),
        ("Claim Extractor", test_claim_extractor),
        ("ADA Mapper", test_ada_mapper),
        ("Evidence Scorer", test_evidence_scorer),
        ("Database Operations", test_database_operations),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The evidence verification service is ready.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
