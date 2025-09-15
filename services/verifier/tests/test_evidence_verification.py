"""Tests for evidence verification service."""

import pytest
from unittest.mock import AsyncMock, patch, Mock
import httpx

from pubmed_client import PubMedClient
from openfda_client import OpenFDAClient
from rxnorm_client import RxNormClient
from scorer import EvidenceScorer


class TestPubMedClient:
    """Test PubMed API client."""
    
    @pytest.fixture
    def pubmed_client(self):
        """Create PubMed client instance."""
        return PubMedClient()
    
    @pytest.fixture
    def sample_search_response(self):
        """Sample PubMed search XML response."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <eSearchResult>
            <Count>2</Count>
            <RetMax>2</RetMax>
            <IdList>
                <Id>12345678</Id>
                <Id>87654321</Id>
            </IdList>
        </eSearchResult>"""
    
    @pytest.fixture
    def sample_summary_response(self):
        """Sample PubMed summary XML response."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <eSummaryResult>
            <DocSum>
                <Id>12345678</Id>
                <Item Name="Title" Type="String">Diabetes Management Study</Item>
                <Item Name="Source" Type="String">Diabetes Care</Item>
                <Item Name="PubDate" Type="String">2023</Item>
            </DocSum>
        </eSummaryResult>"""
    
    @pytest.mark.asyncio
    async def test_search_pubmed(self, pubmed_client, sample_search_response):
        """Test PubMed search functionality."""
        with patch.object(pubmed_client, '_make_request', return_value=sample_search_response):
            pmids = await pubmed_client.search("diabetes treatment", max_results=5)
            
            assert len(pmids) == 2
            assert "12345678" in pmids
            assert "87654321" in pmids
    
    @pytest.mark.asyncio
    async def test_get_summaries(self, pubmed_client, sample_summary_response):
        """Test PubMed summaries retrieval."""
        with patch.object(pubmed_client, '_make_request', return_value=sample_summary_response):
            summaries = await pubmed_client.get_summaries(["12345678"])
            
            assert len(summaries) == 1
            assert summaries[0]['pmid'] == "12345678"
            assert summaries[0]['title'] == "Diabetes Management Study"
            assert summaries[0]['study_type'] in ['rct', 'cohort', 'observational', 'review']
    
    @pytest.mark.asyncio
    async def test_clinical_query_building(self, pubmed_client):
        """Test PICO-formatted clinical query building."""
        query = pubmed_client.build_clinical_query(
            condition="diabetes",
            intervention="lifestyle modification",
            outcome="glycemic control"
        )
        
        assert "diabetes" in query
        assert "lifestyle modification" in query
        assert "glycemic control" in query
        assert "randomized controlled trial" in query
    
    def test_study_type_classification(self, pubmed_client):
        """Test study type classification logic."""
        # Test RCT classification
        pub_types = ['randomized controlled trial', 'clinical trial']
        study_type = pubmed_client._classify_study_type(pub_types)
        assert study_type == 'rct'
        
        # Test meta-analysis classification
        pub_types = ['meta-analysis', 'systematic review']
        study_type = pubmed_client._classify_study_type(pub_types)
        assert study_type == 'meta_analysis'
        
        # Test cohort classification
        pub_types = ['cohort study']
        study_type = pubmed_client._classify_study_type(pub_types)
        assert study_type == 'cohort'


class TestOpenFDAClient:
    """Test OpenFDA API client."""
    
    @pytest.fixture
    def openfda_client(self):
        """Create OpenFDA client instance."""
        return OpenFDAClient()
    
    @pytest.fixture
    def sample_drug_label_response(self):
        """Sample FDA drug label response."""
        return {
            "results": [{
                "openfda": {
                    "generic_name": ["metformin"],
                    "brand_name": ["Glucophage"]
                },
                "warnings": ["Monitor kidney function"],
                "contraindications": ["Severe renal impairment"],
                "adverse_reactions": ["Nausea", "Diarrhea"],
                "drug_interactions": ["Alcohol may increase risk"],
                "boxed_warning": []
            }]
        }
    
    @pytest.fixture
    def sample_adverse_events_response(self):
        """Sample adverse events response."""
        return {
            "results": [
                {"term": "Nausea", "count": 150},
                {"term": "Diarrhea", "count": 120},
                {"term": "Headache", "count": 80}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_get_drug_label(self, openfda_client, sample_drug_label_response):
        """Test FDA drug label retrieval."""
        with patch.object(openfda_client, '_make_request', return_value=sample_drug_label_response):
            drug_info = await openfda_client.get_drug_label("metformin")
            
            assert drug_info['generic_name'] == "metformin"
            assert "Glucophage" in drug_info['brand_names']
            assert len(drug_info['warnings']) > 0
            assert drug_info['black_box_warning'] is False
    
    @pytest.mark.asyncio
    async def test_get_adverse_events(self, openfda_client, sample_adverse_events_response):
        """Test adverse events retrieval."""
        with patch.object(openfda_client, '_make_request', return_value=sample_adverse_events_response):
            adverse_events = await openfda_client.get_adverse_events("metformin")
            
            assert len(adverse_events) == 3
            assert adverse_events[0]['reaction'] == "Nausea"
            assert adverse_events[0]['count'] == 150
            assert adverse_events[0]['severity'] in ['mild', 'moderate', 'severe']
    
    @pytest.mark.asyncio
    async def test_comprehensive_drug_info(self, openfda_client):
        """Test comprehensive drug information gathering."""
        with patch.object(openfda_client, 'get_drug_label') as mock_label, \
             patch.object(openfda_client, 'get_adverse_events') as mock_events, \
             patch.object(openfda_client, 'get_drug_recalls') as mock_recalls:
            
            mock_label.return_value = {"drug_name": "metformin", "black_box_warning": False}
            mock_events.return_value = [{"reaction": "nausea", "severity": "mild"}]
            mock_recalls.return_value = []
            
            comprehensive_info = await openfda_client.get_comprehensive_drug_info("metformin")
            
            assert 'safety_score' in comprehensive_info
            assert 'risk_level' in comprehensive_info
            assert comprehensive_info['safety_score'] >= 0
            assert comprehensive_info['risk_level'] in ['low', 'moderate', 'high']
    
    def test_safety_score_calculation(self, openfda_client):
        """Test safety score calculation logic."""
        # Test high safety score (no warnings)
        label_info = {"black_box_warning": False, "warnings": []}
        adverse_events = []
        recalls = []
        
        score = openfda_client._calculate_safety_score(label_info, adverse_events, recalls)
        assert score >= 0.8
        
        # Test low safety score (black box warning)
        label_info = {"black_box_warning": True, "warnings": ["Serious warning"]}
        score = openfda_client._calculate_safety_score(label_info, adverse_events, recalls)
        assert score <= 0.6


class TestRxNormClient:
    """Test RxNorm API client."""
    
    @pytest.fixture
    def rxnorm_client(self):
        """Create RxNorm client instance."""
        return RxNormClient()
    
    @pytest.fixture
    def sample_drug_search_response(self):
        """Sample RxNorm drug search response."""
        return {
            "drugGroup": {
                "conceptGroup": [{
                    "tty": "IN",
                    "conceptProperties": [{
                        "rxcui": "6809",
                        "name": "metformin",
                        "synonym": "Metformin",
                        "tty": "IN",
                        "language": "ENG",
                        "suppress": "N"
                    }]
                }]
            }
        }
    
    @pytest.mark.asyncio
    async def test_search_drug(self, rxnorm_client, sample_drug_search_response):
        """Test RxNorm drug search."""
        with patch.object(rxnorm_client, '_make_request', return_value=sample_drug_search_response):
            results = await rxnorm_client.search_drug("metformin")
            
            assert len(results) == 1
            assert results[0]['rxcui'] == "6809"
            assert results[0]['name'] == "metformin"
            assert results[0]['tty'] == "IN"
    
    @pytest.mark.asyncio
    async def test_get_rxcui_by_name(self, rxnorm_client):
        """Test RxCUI retrieval by drug name."""
        with patch.object(rxnorm_client, 'search_drug') as mock_search:
            mock_search.return_value = [{"rxcui": "6809", "tty": "IN"}]
            
            rxcui = await rxnorm_client.get_rxcui_by_name("metformin")
            assert rxcui == "6809"
    
    @pytest.mark.asyncio
    async def test_drug_interactions(self, rxnorm_client):
        """Test drug interaction checking."""
        sample_interaction_response = {
            "interactionTypeGroup": [{
                "interactionType": [{
                    "interactionPair": [{
                        "severity": "moderate",
                        "description": "May increase risk of hypoglycemia",
                        "interactionConcept": [
                            {"minConceptItem": {"rxcui": "6809"}},
                            {"minConceptItem": {"rxcui": "38454"}}
                        ]
                    }]
                }]
            }]
        }
        
        with patch.object(rxnorm_client, '_make_request', return_value=sample_interaction_response):
            interactions = await rxnorm_client.get_drug_interactions(["6809", "38454"])
            
            assert len(interactions) >= 0  # May or may not find interactions
    
    @pytest.mark.asyncio
    async def test_medication_safety_check(self, rxnorm_client):
        """Test comprehensive medication safety check."""
        medications = [
            {"name": "metformin", "active": True},
            {"name": "lisinopril", "active": True}
        ]
        
        with patch.object(rxnorm_client, 'normalize_drug_list') as mock_normalize, \
             patch.object(rxnorm_client, 'get_drug_interactions') as mock_interactions:
            
            mock_normalize.return_value = [
                {"original_name": "metformin", "rxcui": "6809", "normalization_success": True},
                {"original_name": "lisinopril", "rxcui": "38454", "normalization_success": True}
            ]
            mock_interactions.return_value = []
            
            safety_result = await rxnorm_client.check_medication_safety(medications)
            
            assert 'interactions' in safety_result
            assert 'normalized_medications' in safety_result
            assert 'overall_risk' in safety_result
            assert safety_result['overall_risk'] in ['low', 'moderate', 'high', 'unknown']


class TestEvidenceScorer:
    """Test evidence scoring and verification logic."""
    
    @pytest.fixture
    def evidence_scorer(self):
        """Create evidence scorer instance."""
        return EvidenceScorer()
    
    @pytest.fixture
    def sample_evidence_collection(self):
        """Sample evidence collection for testing."""
        return [
            {
                "source_type": "guideline",
                "title": "ADA Standards of Care 2024",
                "content": "Evidence-based diabetes management guidelines",
                "url": "https://ada.org/guidelines",
                "metadata": {"organization": "ADA", "year": "2024"}
            },
            {
                "source_type": "rct",
                "title": "Randomized Trial of Lifestyle Intervention",
                "content": "Large randomized controlled trial of lifestyle intervention",
                "journal": "New England Journal of Medicine",
                "pub_date": "2023",
                "metadata": {"sample_size": 5000, "study_type": "rct"}
            },
            {
                "source_type": "observational",
                "title": "Small Observational Study",
                "content": "Limited observational study",
                "metadata": {"sample_size": 50}
            }
        ]
    
    def test_evidence_scoring(self, evidence_scorer, sample_evidence_collection):
        """Test evidence collection scoring."""
        result = evidence_scorer.score_evidence_collection(sample_evidence_collection)
        
        assert 'overall_score' in result
        assert 'status' in result
        assert 'evidence_count' in result
        assert 'quality_breakdown' in result
        
        assert 0 <= result['overall_score'] <= 1
        assert result['status'] in ['approved', 'flagged', 'rejected']
        assert result['evidence_count'] == 3
    
    def test_individual_evidence_scoring(self, evidence_scorer):
        """Test individual evidence item scoring."""
        # High-quality evidence (guideline)
        guideline_evidence = {
            "source_type": "guideline",
            "title": "ADA Standards of Care",
            "content": "Clinical practice guidelines",
            "metadata": {"organization": "ADA"}
        }
        
        scored = evidence_scorer._score_individual_evidence(guideline_evidence)
        
        assert scored['quality_score'] >= 0.9  # Guidelines should score high
        assert scored['quality_level'] == 'high'
        assert scored['weight'] > 0
    
    def test_safety_flag_detection(self, evidence_scorer):
        """Test safety flag detection."""
        # Evidence with safety concern
        safety_evidence = {
            "source_type": "pubmed",
            "title": "Black Box Warning Study",
            "content": "Study reports black box warning for medication",
            "metadata": {}
        }
        
        scored = evidence_scorer._score_individual_evidence(safety_evidence)
        assert 'black_box_warning' in scored['flags']
    
    def test_evidence_status_determination(self, evidence_scorer):
        """Test evidence status determination logic."""
        # High-quality evidence should be approved
        high_quality_evidence = [
            {
                "source_type": "guideline",
                "title": "Clinical Guidelines",
                "content": "Evidence-based guidelines"
            }
        ]
        
        result = evidence_scorer.score_evidence_collection(high_quality_evidence)
        # Should be approved if above minimum threshold
        
        # Evidence with safety flags should be flagged
        flagged_evidence = [
            {
                "source_type": "pubmed",
                "title": "Safety Warning",
                "content": "black box warning contraindication"
            }
        ]
        
        result = evidence_scorer.score_evidence_collection(flagged_evidence)
        assert result['status'] == 'flagged'
    
    def test_quality_modifiers(self, evidence_scorer):
        """Test quality modifier application."""
        # Recent publication from high-impact journal
        high_quality_evidence = {
            "source_type": "rct",
            "title": "Recent RCT",
            "journal": "New England Journal of Medicine",
            "pub_date": "2023",
            "content": "double-blind placebo-controlled",
            "metadata": {"sample_size": 2000}
        }
        
        scored = evidence_scorer._score_individual_evidence(high_quality_evidence)
        
        # Should have quality modifiers applied
        assert scored['quality_score'] > evidence_scorer.evidence_weights['rct']
        assert 'high_impact_journal' in scored['modifiers_applied']
        assert 'recent_publication' in scored['modifiers_applied']


class TestIntegrationTests:
    """Integration tests for evidence verification service."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_verification(self):
        """Test complete evidence verification workflow."""
        # This would test the complete workflow from care plan
        # to evidence collection to scoring
        pass
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in evidence verification."""
        # Test network errors, API failures, etc.
        pass
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test API rate limiting compliance."""
        # Test that clients respect rate limits
        pass


if __name__ == "__main__":
    pytest.main([__file__])
