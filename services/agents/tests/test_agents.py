"""Tests for GenAI agents system."""

import pytest
from unittest.mock import AsyncMock, patch, Mock
from datetime import datetime

from base_agent import AgentContext, BaseAgent, MockLLMAgent
from intake_agent import IntakeAgent
from normalizer_agent import NormalizerAgent
from orchestrator_agent import OrchestratorAgent


class TestAgentContext:
    """Test agent context management."""
    
    def test_context_creation(self):
        """Test agent context creation."""
        patient_id = "test-patient-123"
        context = AgentContext(patient_id)
        
        assert context.patient_id == patient_id
        assert context.snapshot_ts is not None
        assert len(context.agent_outputs) == 0
        assert len(context.errors) == 0
    
    def test_context_data_management(self):
        """Test context data management."""
        context = AgentContext("test-patient")
        
        # Add patient data
        demographics = {"age": 45, "sex": "F"}
        vitals = [{"type": "glucose", "value": 100}]
        medications = [{"name": "metformin"}]
        conditions = [{"name": "diabetes"}]
        
        context.add_patient_data(demographics, vitals, medications, conditions)
        
        assert context.patient_profile == demographics
        assert context.vitals == vitals
        assert context.medications == medications
        assert context.conditions == conditions
    
    def test_agent_output_management(self):
        """Test agent output management."""
        context = AgentContext("test-patient")
        
        # Add agent output
        output = {"processed": True, "score": 0.85}
        context.add_agent_output("test_agent", output)
        
        assert context.get_agent_output("test_agent") == output
        assert context.get_agent_output("nonexistent_agent") is None
    
    def test_error_management(self):
        """Test error management."""
        context = AgentContext("test-patient")
        
        # Add error
        context.add_error("test_agent", "Test error message")
        
        assert len(context.errors) == 1
        assert context.errors[0]["agent"] == "test_agent"
        assert context.errors[0]["error"] == "Test error message"
    
    def test_context_serialization(self):
        """Test context to/from dict conversion."""
        context = AgentContext("test-patient")
        context.add_patient_data({"age": 45}, [], [], [])
        context.add_agent_output("test", {"result": True})
        
        # Convert to dict
        context_dict = context.to_dict()
        assert isinstance(context_dict, dict)
        assert context_dict["patient_id"] == "test-patient"
        
        # Convert back from dict
        restored_context = AgentContext.from_dict(context_dict)
        assert restored_context.patient_id == context.patient_id
        assert restored_context.patient_profile == context.patient_profile


class TestBaseAgent:
    """Test base agent functionality."""
    
    def test_base_agent_creation(self):
        """Test base agent creation."""
        class TestAgent(BaseAgent):
            async def process(self, context):
                return context
        
        agent = TestAgent("test_agent", "Test agent description")
        assert agent.name == "test_agent"
        assert agent.description == "Test agent description"
    
    def test_context_validation(self):
        """Test context validation."""
        class TestAgent(BaseAgent):
            async def process(self, context):
                return context
        
        agent = TestAgent("test_agent")
        context = AgentContext("test-patient")
        
        # Valid context
        assert agent.validate_context(context, ["patient_id"])
        
        # Invalid context (missing required field)
        assert not agent.validate_context(context, ["nonexistent_field"])
    
    def test_error_handling(self):
        """Test agent error handling."""
        class TestAgent(BaseAgent):
            async def process(self, context):
                return context
        
        agent = TestAgent("test_agent")
        context = AgentContext("test-patient")
        
        # Handle error
        test_error = Exception("Test error")
        updated_context = agent.handle_error(context, test_error)
        
        assert len(updated_context.errors) == 1
        assert updated_context.errors[0]["agent"] == "test_agent"


class TestMockLLMAgent:
    """Test mock LLM agent for development."""
    
    @pytest.mark.asyncio
    async def test_mock_llm_responses(self):
        """Test mock LLM response generation."""
        agent = MockLLMAgent("mock_agent")
        
        # Test risk-related response
        risk_response = await agent.call_llm("patient risk assessment")
        assert "risk" in risk_response.lower()
        
        # Test care plan response
        care_response = await agent.call_llm("generate care plan recommendations")
        assert "care plan" in care_response.lower() or "recommendation" in care_response.lower()
    
    def test_mock_response_parsing(self):
        """Test mock response parsing."""
        agent = MockLLMAgent("mock_agent")
        
        # Test JSON parsing
        json_response = '{"test": "value", "score": 0.8}'
        parsed = agent.parse_llm_response(json_response, "json")
        
        assert isinstance(parsed, dict)
        assert parsed["test"] == "value"
        assert parsed["score"] == 0.8


class TestIntakeAgent:
    """Test intake agent functionality."""
    
    @pytest.fixture
    def intake_agent(self):
        """Create intake agent."""
        return IntakeAgent()
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing."""
        return {
            "demographics": {
                "age": 55,
                "sex": "F",
                "bmi": 28.5,
                "weight_kg": 75,
                "height_cm": 165
            },
            "vitals": [
                {"type": "glucose_fasting", "value": 120, "unit": "mg/dL", "ts": "2024-09-15T08:00:00Z"},
                {"type": "hba1c", "value": 7.2, "unit": "%", "ts": "2024-09-15T10:00:00Z"},
                {"type": "sbp", "value": 140, "unit": "mmHg", "ts": "2024-09-15T09:00:00Z"}
            ],
            "medications": [
                {"name": "Metformin", "dosage": "500mg", "active": True, "rxnorm_code": "6809"}
            ],
            "conditions": [
                {"name": "Type 2 Diabetes Mellitus", "icd10_code": "E11.9", "active": True, "severity": "moderate"}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_intake_processing(self, intake_agent, sample_patient_data):
        """Test intake agent processing."""
        context = AgentContext("test-patient")
        context.add_patient_data(
            sample_patient_data["demographics"],
            sample_patient_data["vitals"],
            sample_patient_data["medications"],
            sample_patient_data["conditions"]
        )
        
        result_context = await intake_agent.process(context)
        
        # Check that intake output was added
        intake_output = result_context.get_agent_output("intake")
        assert intake_output is not None
        assert "validation_status" in intake_output
        assert "data_completeness" in intake_output
        assert intake_output["vitals_count"] == 3
        assert intake_output["medications_count"] == 1
    
    def test_demographics_validation(self, intake_agent):
        """Test demographics validation."""
        # Valid demographics
        valid_demographics = {"age": 45, "sex": "F", "bmi": 25.0}
        issues = intake_agent._validate_demographics(valid_demographics)
        assert len(issues) == 0
        
        # Invalid demographics
        invalid_demographics = {"age": -5, "sex": "X", "bmi": 100}
        issues = intake_agent._validate_demographics(invalid_demographics)
        assert len(issues) > 0
    
    def test_vitals_validation(self, intake_agent):
        """Test vitals validation."""
        # Valid vitals
        valid_vitals = [
            {"type": "glucose_fasting", "value": 100, "unit": "mg/dL", "ts": "2024-09-15T08:00:00Z"}
        ]
        issues = intake_agent._validate_vitals(valid_vitals)
        assert len(issues) == 0
        
        # Invalid vitals
        invalid_vitals = [
            {"type": "glucose_fasting", "value": 1000, "unit": "mg/dL"},  # Unrealistic value
            {"value": 100}  # Missing type
        ]
        issues = intake_agent._validate_vitals(invalid_vitals)
        assert len(issues) > 0
    
    def test_data_completeness_assessment(self, intake_agent):
        """Test data completeness assessment."""
        context = AgentContext("test-patient")
        context.patient_profile = {"age": 45, "sex": "F", "bmi": 25.0}
        context.vitals = [{"type": "glucose_fasting", "value": 100}]
        
        completeness = intake_agent._assess_data_completeness(context)
        
        assert "demographics_completeness" in completeness
        assert "vitals_completeness" in completeness
        assert "overall_completeness" in completeness
        assert 0 <= completeness["overall_completeness"] <= 1


class TestNormalizerAgent:
    """Test normalizer agent functionality."""
    
    @pytest.fixture
    def normalizer_agent(self):
        """Create normalizer agent."""
        return NormalizerAgent()
    
    @pytest.mark.asyncio
    async def test_normalization_processing(self, normalizer_agent):
        """Test normalizer agent processing."""
        context = AgentContext("test-patient")
        context.patient_profile = {"age": 45, "sex": "FEMALE", "weight": 150, "weight_unit": "lb"}
        context.vitals = [
            {"type": "glucose", "value": 6.7, "unit": "mmol/L"},  # Should convert to mg/dL
            {"type": "blood_pressure_systolic", "value": 140, "unit": "mmHg"}
        ]
        
        result_context = await normalizer_agent.process(context)
        
        # Check normalization output
        norm_output = result_context.get_agent_output("normalizer")
        assert norm_output is not None
        assert norm_output["demographics_normalized"] is True
        
        # Check normalized data
        assert result_context.patient_profile["sex"] == "F"  # Normalized from "FEMALE"
        assert result_context.patient_profile["weight_kg"] == pytest.approx(68.04, rel=1e-2)  # Converted from lb
    
    def test_unit_conversions(self, normalizer_agent):
        """Test unit conversion functionality."""
        # Test weight conversion (lb to kg)
        converted = normalizer_agent._convert_unit(150, "lb", "weight")
        assert converted == pytest.approx(68.04, rel=1e-2)
        
        # Test glucose conversion (mmol/L to mg/dL)
        converted = normalizer_agent._convert_unit(6.7, "mmol/L", "glucose")
        assert converted == pytest.approx(120.7, rel=1e-1)
        
        # Test no conversion needed
        converted = normalizer_agent._convert_unit(100, "mg/dL", "glucose")
        assert converted == 100
    
    def test_vital_type_normalization(self, normalizer_agent):
        """Test vital type normalization."""
        # Test common variations
        assert normalizer_agent._normalize_vital_type("glucose") == "glucose_fasting"
        assert normalizer_agent._normalize_vital_type("systolic") == "sbp"
        assert normalizer_agent._normalize_vital_type("diastolic") == "dbp"
        assert normalizer_agent._normalize_vital_type("heart_rate") == "heart_rate"
        assert normalizer_agent._normalize_vital_type("unknown_type") == "unknown_type"
    
    def test_medication_normalization(self, normalizer_agent):
        """Test medication name normalization."""
        # Test common standardizations
        assert normalizer_agent._normalize_medication_name("metformin hcl") == "Metformin"
        assert normalizer_agent._normalize_medication_name("ATORVASTATIN CALCIUM") == "Atorvastatin"
    
    def test_bmi_calculation(self, normalizer_agent):
        """Test BMI calculation during normalization."""
        demographics = {
            "weight_kg": 70,
            "height_cm": 170,
            "bmi": 20.0  # Existing BMI that should be recalculated
        }
        
        normalized = normalizer_agent._normalize_demographics(demographics)
        expected_bmi = 70 / (1.7 ** 2)  # Should be ~24.2
        
        assert normalized["bmi"] == pytest.approx(expected_bmi, rel=1e-1)


class TestOrchestratorAgent:
    """Test orchestrator agent functionality."""
    
    @pytest.fixture
    def orchestrator_agent(self):
        """Create orchestrator agent."""
        return OrchestratorAgent()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator_agent):
        """Test orchestrator initialization."""
        assert orchestrator_agent.intake_agent is not None
        assert orchestrator_agent.normalizer_agent is not None
        assert orchestrator_agent.risk_predictor_agent is not None
        assert orchestrator_agent.careplan_agent is not None
        assert orchestrator_agent.evidence_verifier_agent is not None
    
    @pytest.mark.asyncio
    @patch('orchestrator_agent.OrchestratorAgent._load_patient_data')
    @patch('orchestrator_agent.OrchestratorAgent._store_recommendation')
    @patch('orchestrator_agent.OrchestratorAgent._create_audit_log')
    async def test_recommendation_generation(self, mock_audit, mock_store, mock_load, orchestrator_agent):
        """Test complete recommendation generation."""
        # Mock dependencies
        mock_load.return_value = None  # Load patient data
        mock_store.return_value = "test-recommendation-id"
        mock_audit.return_value = None
        
        # Mock agent processing
        with patch.object(orchestrator_agent.intake_agent, 'process') as mock_intake, \
             patch.object(orchestrator_agent.normalizer_agent, 'process') as mock_normalizer, \
             patch.object(orchestrator_agent.risk_predictor_agent, 'process') as mock_risk, \
             patch.object(orchestrator_agent.careplan_agent, 'process') as mock_careplan, \
             patch.object(orchestrator_agent.evidence_verifier_agent, 'process') as mock_evidence:
            
            # Setup mock returns
            mock_intake.return_value = AgentContext("test-patient")
            mock_normalizer.return_value = AgentContext("test-patient")
            
            risk_context = AgentContext("test-patient")
            risk_context.add_agent_output("risk_predictor", {
                "risk_probability": 0.25,
                "risk_category": "moderate",
                "model_version": "test_v1.0"
            })
            mock_risk.return_value = risk_context
            
            careplan_context = AgentContext("test-patient")
            careplan_context.add_agent_output("careplan_generator", {
                "care_plan": {"dietary": {"recommendations": ["Reduce carbs"]}}
            })
            mock_careplan.return_value = careplan_context
            
            evidence_context = AgentContext("test-patient")
            evidence_context.add_agent_output("evidence_verifier", {
                "overall_score": 0.8,
                "status": "approved",
                "evidence_links": []
            })
            mock_evidence.return_value = evidence_context
            
            # Test recommendation generation
            result = await orchestrator_agent.generate_recommendation("test-patient")
            
            assert result["success"] is True
            assert "recommendation_id" in result
            assert "recommendation" in result
    
    def test_evidence_strength_assessment(self, orchestrator_agent):
        """Test evidence strength assessment."""
        # High score
        strength = orchestrator_agent._assess_evidence_strength({"overall_score": 0.9})
        assert "High" in strength
        
        # Moderate score
        strength = orchestrator_agent._assess_evidence_strength({"overall_score": 0.7})
        assert "Moderate" in strength
        
        # Low score
        strength = orchestrator_agent._assess_evidence_strength({"overall_score": 0.3})
        assert "Low" in strength or "Very Low" in strength


class TestErrorHandling:
    """Test error handling across agents."""
    
    @pytest.mark.asyncio
    async def test_agent_error_propagation(self):
        """Test that agent errors are properly handled and propagated."""
        class FailingAgent(BaseAgent):
            async def process(self, context):
                raise Exception("Test agent failure")
        
        agent = FailingAgent("failing_agent")
        context = AgentContext("test-patient")
        
        result_context = await agent.process(context)
        
        # Should have error in context
        assert len(result_context.errors) == 1
        assert result_context.errors[0]["agent"] == "failing_agent"
    
    @pytest.mark.asyncio
    async def test_orchestrator_error_resilience(self):
        """Test orchestrator resilience to individual agent failures."""
        # This would test that orchestrator can continue even if one agent fails
        pass


class TestPerformance:
    """Test agent performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_agent_processing_time(self):
        """Test that agents process within reasonable time limits."""
        # This would test performance benchmarks
        pass
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test agent memory usage."""
        # This would test memory efficiency
        pass


if __name__ == "__main__":
    pytest.main([__file__])
