"""Test script for evidence verification service."""

import asyncio
import json
from typing import Dict, Any
import structlog

from .service import create_verification_service
from .models import VerificationRequest
from .config import get_default_config

logger = structlog.get_logger()


async def test_evidence_verification():
    """Test the evidence verification service."""
    
    # Sample care plan
    care_plan = {
        "exercise": {
            "aerobic": "30 minutes of moderate-intensity aerobic activity 5 days per week",
            "resistance": "Resistance training 2-3 times per week",
            "monitoring": "Monitor heart rate during exercise"
        },
        "dietary": {
            "recommendations": [
                "Reduce carbohydrate intake to 45-60g per meal",
                "Increase fiber intake to 25-35g per day",
                "Limit sodium to 2300mg per day"
            ]
        },
        "medication_safety": {
            "current_regimen": "Metformin 1000mg twice daily",
            "monitoring": "Monitor HbA1c every 3 months",
            "warnings": ["Avoid sulfonylureas due to hypoglycemia risk"]
        }
    }
    
    # Sample patient context
    patient_context = {
        "age": 55,
        "sex": "male",
        "conditions": ["type 2 diabetes", "hypertension"],
        "medications": ["metformin", "lisinopril"],
        "ckd": False,
        "pregnancy": False,
        "comorbidities": ["obesity"]
    }
    
    # Create verification request
    request = VerificationRequest(
        recommendation_id="test-rec-001",
        care_plan=care_plan,
        patient_context=patient_context,
        max_evidence_per_claim=10,
        include_external_apis=True
    )
    
    print("Testing Evidence Verification Service")
    print("=" * 50)
    
    try:
        # Create mock database pool (for testing)
        import asyncpg
        database_url = "postgresql://user:password@localhost:5432/hackwell"
        
        # Note: In a real test, you would use a test database
        print("Note: This test requires a running PostgreSQL database")
        print("Database URL:", database_url)
        
        # Create database pool
        db_pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=5
        )
        
        # Get configuration
        config = get_default_config()
        config.update({
            "openai_api_key": "your-openai-api-key",
            "pubmed_email": "your-email@example.com"
        })
        
        # Create verification service
        service = create_verification_service(db_pool, config)
        
        print("\n1. Testing claim extraction...")
        claims = service.claim_extractor.extract_claims(
            care_plan, 
            service.claim_extractor.ClaimContext(**patient_context)
        )
        print(f"   Extracted {len(claims)} claims:")
        for claim in claims:
            print(f"   - {claim.id}: {claim.text}")
        
        print("\n2. Testing evidence verification...")
        result = await service.verify_recommendation(request)
        
        print(f"\nVerification Result:")
        print(f"  Recommendation ID: {result['recommendation_id']}")
        print(f"  Overall Status: {result['overall_status']}")
        print(f"  Total Evidence: {result['total_evidence']}")
        print(f"  Supporting Evidence: {result['supporting_evidence']}")
        print(f"  Contradicting Evidence: {result['contradicting_evidence']}")
        print(f"  Warning Evidence: {result['warning_evidence']}")
        
        print(f"\nClaims ({len(result['claims'])}):")
        for i, claim in enumerate(result['claims'], 1):
            print(f"  {i}. {claim['claim_text']}")
            print(f"     Verdict: {claim['verdict']}")
            print(f"     Support Score: {claim['support_score']:.3f}")
            print(f"     Contradict Score: {claim['contradict_score']:.3f}")
            print(f"     Evidence Items: {len(claim['items'])}")
            
            # Show top evidence items
            if claim['items']:
                print("     Top Evidence:")
                for j, item in enumerate(claim['items'][:3], 1):
                    print(f"       {j}. [{item['source_type']}] {item['stance']} (score: {item['score']:.3f})")
                    print(f"          {item['snippet'][:100]}...")
        
        # Close database pool
        await db_pool.close()
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        logger.error("Test failed", error=str(e))


async def test_individual_components():
    """Test individual components."""
    
    print("\nTesting Individual Components")
    print("=" * 50)
    
    try:
        # Test claim extractor
        print("\n1. Testing Claim Extractor...")
        from .claim_extractor import create_claim_extractor, ClaimContext
        
        claim_extractor = create_claim_extractor()
        patient_context = ClaimContext(
            age=55,
            conditions=["type 2 diabetes"],
            medications=["metformin"]
        )
        
        care_plan = {
            "exercise": {
                "aerobic": "30 minutes of moderate-intensity aerobic activity 5 days per week"
            }
        }
        
        claims = claim_extractor.extract_claims(care_plan, patient_context)
        print(f"   Extracted {len(claims)} claims")
        for claim in claims:
            print(f"   - {claim.text}")
        
        # Test stance classifier (mock)
        print("\n2. Testing Stance Classifier...")
        from .stance import create_stance_classifier
        
        stance_classifier = create_stance_classifier(classifier_type="mock")
        
        claim_text = "Exercise improves glycemic control in diabetes"
        passage_text = "Regular physical activity has been shown to improve HbA1c levels in patients with type 2 diabetes."
        
        stance_result = await stance_classifier.classify(claim_text, passage_text)
        print(f"   Claim: {claim_text}")
        print(f"   Passage: {passage_text}")
        print(f"   Stance: {stance_result.stance} (confidence: {stance_result.confidence:.3f})")
        
        # Test evidence scorer
        print("\n3. Testing Evidence Scorer...")
        from .scorer import create_evidence_scorer
        
        evidence_scorer = create_evidence_scorer()
        
        score = evidence_scorer.score_individual_evidence(
            content=passage_text,
            source_type="rct",
            stance=stance_result.stance,
            stance_confidence=stance_result.confidence,
            retrieval_score=0.8,
            pub_date="2023-01-01",
            quality=0.9
        )
        print(f"   Evidence Score: {score:.3f}")
        
        # Test ADA mapper
        print("\n4. Testing ADA Mapper...")
        from .adapters.ada_mapper import create_ada_mapper
        
        ada_mapper = create_ada_mapper()
        statements = ada_mapper.search_statements("exercise diabetes", max_results=3)
        print(f"   Found {len(statements)} ADA statements")
        for statement in statements:
            print(f"   - {statement.title}")
        
        print("\n✅ Component tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Component test failed: {str(e)}")
        logger.error("Component test failed", error=str(e))


async def main():
    """Main test function."""
    print("Evidence Verification Service Test Suite")
    print("=" * 60)
    
    # Test individual components first
    await test_individual_components()
    
    # Test full verification (requires database)
    print("\n" + "=" * 60)
    print("Full Verification Test (requires database setup)")
    print("=" * 60)
    
    # Uncomment to run full test
    # await test_evidence_verification()


if __name__ == "__main__":
    asyncio.run(main())
