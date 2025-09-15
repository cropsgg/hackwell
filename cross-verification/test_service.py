#!/usr/bin/env python3
"""Test script for evidence verification service components."""

import asyncio
import json
from datetime import datetime
from uuid import uuid4

async def test_verification_pipeline():
    """Test the core verification pipeline components."""
    print("🧪 Testing Evidence Verification Pipeline")
    print("=" * 50)
    
    # Test 1: Database Connection
    print("\n1. Testing Database Connection...")
    try:
        import asyncpg
        conn = await asyncpg.connect("postgresql://postgres:password@localhost:5432/hackwell")
        
        # Test basic query
        result = await conn.fetchval("SELECT COUNT(*) FROM evidence_documents")
        print(f"   ✅ Database connected - {result} evidence documents found")
        
        # Test hybrid search function
        search_result = await conn.fetch("""
            SELECT * FROM hybrid_search(
                'diabetes exercise',
                ARRAY_FILL(0.1, ARRAY[1536])::vector(1536),
                5,
                5
            ) LIMIT 3
        """)
        print(f"   ✅ Hybrid search working - {len(search_result)} results")
        
        await conn.close()
        
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False
    
    # Test 2: Embeddings Provider
    print("\n2. Testing Embeddings Provider...")
    try:
        from embeddings import create_embeddings_provider
        
        # Use mock provider to avoid API calls
        provider = create_embeddings_provider("mock", dimension=1536)
        result = await provider.embed_text("Test text for embedding")
        print(f"   ✅ Mock embeddings working - {len(result.embedding)} dimensions")
        
    except Exception as e:
        print(f"   ❌ Embeddings test failed: {e}")
        return False
    
    # Test 3: Claim Extractor
    print("\n3. Testing Claim Extractor...")
    try:
        from claim_extractor import create_claim_extractor
        
        extractor = create_claim_extractor()
        
        # Test recommendation
        recommendation = {
            "id": str(uuid4()),
            "type": "medication",
            "title": "Start Metformin",
            "description": "Patient should start metformin 500mg twice daily for diabetes management",
            "medication": "metformin",
            "dosage": "500mg twice daily",
            "conditions": ["type 2 diabetes"],
            "outcomes": ["blood glucose control"]
        }
        
        from claim_extractor import ClaimContext
        patient_context = ClaimContext(
            age=45,
            conditions=["type 2 diabetes"],
            medications=[],
            allergies=[],
            lab_values={}
        )
        claims = extractor.extract_claims(recommendation, patient_context)
        print(f"   ✅ Claim extraction working - {len(claims)} claims extracted")
        for i, claim in enumerate(claims, 1):
            print(f"      Claim {i}: {claim.claim_text}")
        
    except Exception as e:
        print(f"   ❌ Claim extraction test failed: {e}")
        return False
    
    # Test 4: Mock Stance Classification
    print("\n4. Testing Stance Classification...")
    try:
        from stance import create_stance_classifier
        
        # Use mock classifier to avoid model loading
        classifier = create_stance_classifier("mock")
        
        result = await classifier.classify(
            "Metformin is effective for diabetes",
            "Metformin significantly reduces HbA1c levels in patients with type 2 diabetes"
        )
        print(f"   ✅ Mock stance classification working - {result.stance} (confidence: {result.confidence:.2f})")
        
    except Exception as e:
        print(f"   ❌ Stance classification test failed: {e}")
        return False
    
    # Test 5: Evidence Scorer
    print("\n5. Testing Evidence Scorer...")
    try:
        from scorer import create_evidence_scorer
        
        scorer = create_evidence_scorer()
        
        # Test evidence item
        evidence_item = {
            "content": "Metformin is the first-line treatment for type 2 diabetes",
            "source_type": "guideline",
            "url": "https://example.com",
            "title": "ADA Guidelines",
            "pub_date": datetime.now(),
            "quality": 0.95
        }
        
        from models import EvidenceItem
        evidence_obj = EvidenceItem(
            source_type=evidence_item["source_type"],
            url=evidence_item["url"],
            title=evidence_item["title"],
            stance="support",
            score=0.8,
            snippet=evidence_item["content"][:100]
        )
        score = scorer.score_individual_evidence(
            evidence_item["content"],
            evidence_item["source_type"],
            "support",
            0.9,
            0.8,
            evidence_item["pub_date"].isoformat() if evidence_item["pub_date"] else None,
            evidence_item["quality"]
        )
        print(f"   ✅ Evidence scoring working - score: {score:.3f}")
        
    except Exception as e:
        print(f"   ❌ Evidence scoring test failed: {e}")
        return False
    
    print("\n🎉 All core components are working correctly!")
    return True

async def test_full_verification():
    """Test a complete verification workflow."""
    print("\n" + "=" * 50)
    print("🔍 Testing Full Verification Workflow")
    print("=" * 50)
    
    try:
        # Create a mock verification service
        from service import create_verification_service
        import asyncpg
        
        # Connect to database
        db_pool = await asyncpg.create_pool(
            "postgresql://postgres:password@localhost:5432/hackwell",
            min_size=1,
            max_size=5
        )
        
        # Create service with mock components
        config = {
            "embeddings_provider": "mock",
            "stance_classifier": "mock",
            "k_semantic": 5,
            "k_lexical": 5,
            "enable_pubmed": False,
            "enable_openfda": False,
            "enable_rxnorm": False,
            "enable_ada": False
        }
        
        service = create_verification_service(db_pool, config)
        
        # Test verification request
        from service import VerificationRequest
        request = VerificationRequest(
            recommendation_id=str(uuid4()),
            care_plan={
                "type": "medication",
                "title": "Start Metformin",
                "description": "Patient should start metformin 500mg twice daily for diabetes management",
                "medication": "metformin",
                "dosage": "500mg twice daily",
                "conditions": ["type 2 diabetes"],
                "outcomes": ["blood glucose control"]
            },
            patient_context={
                "age": 45,
                "conditions": ["type 2 diabetes"],
                "medications": []
            }
        )
        
        print("   Testing verification request...")
        result = await service.verify_recommendation(request)
        
        print(f"   ✅ Verification completed!")
        print(f"      Overall verdict: {result.get('overall_verdict', 'unknown')}")
        print(f"      Claims processed: {len(result.get('claims', []))}")
        print(f"      Evidence found: {len(result.get('evidence', []))}")
        
        # Close database pool
        await db_pool.close()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Full verification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Evidence Verification Service - Component Testing")
    print("=" * 60)
    
    # Test core components
    components_ok = await test_verification_pipeline()
    
    if components_ok:
        # Test full workflow
        workflow_ok = await test_full_verification()
        
        if workflow_ok:
            print("\n🎉 ALL TESTS PASSED! The evidence verification service is working correctly.")
        else:
            print("\n⚠️  Core components work, but full workflow needs attention.")
    else:
        print("\n❌ Some core components failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
