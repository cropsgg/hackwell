#!/usr/bin/env python3
"""Basic test script for evidence verification service."""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_database_connection():
    """Test database connection."""
    try:
        import asyncpg
        
        # Database connection
        database_url = "postgresql://postgres:password@localhost:5432/hackwell"
        conn = await asyncpg.connect(database_url)
        
        # Test basic query
        result = await conn.fetchval("SELECT 1")
        print(f"✅ Database connection successful: {result}")
        
        # Test evidence tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('evidence_documents', 'evidence_chunks', 'evidence_links', 'audit_logs')
            ORDER BY table_name
        """)
        
        print(f"✅ Found {len(tables)} evidence tables:")
        for table in tables:
            print(f"   - {table['table_name']}")
        
        # Test pgvector extension
        extensions = await conn.fetch("""
            SELECT extname 
            FROM pg_extension 
            WHERE extname = 'vector'
        """)
        
        if extensions:
            print("✅ pgvector extension is installed")
        else:
            print("❌ pgvector extension not found")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

async def test_embeddings():
    """Test OpenAI embeddings."""
    try:
        from embeddings import create_embeddings_provider
        
        # Create embeddings provider
        provider = create_embeddings_provider(
            provider_type="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-large"
        )
        
        # Test embedding generation
        result = await provider.embed_text("Test text for embedding")
        print(f"✅ Embeddings working: {len(result.embedding)} dimensions")
        print(f"   Model: {result.model}")
        print(f"   Tokens: {result.usage_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False

async def test_claim_extractor():
    """Test claim extraction."""
    try:
        from claim_extractor import create_claim_extractor, ClaimContext
        
        # Create claim extractor
        extractor = create_claim_extractor()
        
        # Test data
        care_plan = {
            "exercise": {
                "aerobic": "30 minutes of moderate-intensity aerobic activity 5 days per week"
            },
            "medication_safety": {
                "warnings": ["Avoid sulfonylureas due to hypoglycemia risk"]
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
        
        # Test breakdown
        breakdown = scorer.get_scoring_breakdown(
            source_type="rct",
            stance="support",
            stance_confidence=0.8,
            retrieval_score=0.7,
            pub_date="2023-01-01",
            quality=0.9
        )
        
        print(f"   Breakdown: {breakdown}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evidence scoring test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Evidence Verification Service - Basic Tests")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    tests = [
        ("Database Connection", test_database_connection),
        ("OpenAI Embeddings", test_embeddings),
        ("Claim Extractor", test_claim_extractor),
        ("ADA Mapper", test_ada_mapper),
        ("Evidence Scorer", test_evidence_scorer),
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
