"""Tests for patient management endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import uuid

from app.main import app

client = TestClient(app)


class TestPatientEndpoints:
    """Test patient-related API endpoints."""
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        return {
            "id": str(uuid.uuid4()),
            "email": "test@example.com",
            "roles": ["patient"],
            "metadata": {"patient_id": str(uuid.uuid4())}
        }
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing."""
        return {
            "demographics": {
                "name": "Test Patient",
                "age": 45,
                "sex": "F",
                "ethnicity": "Caucasian",
                "height_cm": 165,
                "weight_kg": 70,
                "bmi": 25.7,
                "family_history": {
                    "diabetes": True,
                    "heart_disease": False
                },
                "lifestyle": {
                    "smoking": "never",
                    "alcohol": "occasional",
                    "exercise_mins_week": 150
                }
            },
            "consent": {
                "ehr_access": True,
                "data_sharing": False,
                "research_participation": True
            }
        }
    
    @pytest.fixture
    def sample_vital_data(self):
        """Sample vital signs data."""
        return {
            "type": "glucose_fasting",
            "value": 95.0,
            "unit": "mg/dL",
            "source": "device",
            "ts": "2024-09-15T08:00:00Z"
        }
    
    @patch('app.routers.patients.get_current_active_user')
    @patch('app.routers.patients.db_manager.execute_query')
    def test_create_patient_success(self, mock_db, mock_auth, mock_user, sample_patient_data):
        """Test successful patient creation."""
        mock_auth.return_value = mock_user
        mock_db.return_value = [{
            'id': str(uuid.uuid4()),
            'user_id': mock_user['id'],
            'demographics': sample_patient_data['demographics'],
            'consent': sample_patient_data['consent'],
            'created_at': '2024-09-15T10:00:00Z',
            'updated_at': '2024-09-15T10:00:00Z'
        }]
        
        response = client.post("/api/v1/patients", json=sample_patient_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'data' in data
        assert data['data']['demographics']['name'] == "Test Patient"
    
    @patch('app.routers.patients.get_current_active_user')
    def test_create_patient_unauthorized(self, mock_auth):
        """Test patient creation with insufficient permissions."""
        mock_auth.return_value = {
            "id": str(uuid.uuid4()),
            "roles": [],  # No roles
            "metadata": {}
        }
        
        response = client.post("/api/v1/patients", json={})
        
        assert response.status_code == 403
    
    @patch('app.routers.patients.require_patient_or_clinician_access')
    @patch('app.routers.patients.db_manager.execute_query')
    def test_get_patient_success(self, mock_db, mock_auth, mock_user):
        """Test successful patient retrieval."""
        patient_id = str(uuid.uuid4())
        mock_auth.return_value = mock_user
        mock_db.return_value = [{
            'id': patient_id,
            'user_id': mock_user['id'],
            'demographics': {"name": "Test Patient", "age": 45},
            'consent': {"ehr_access": True},
            'created_at': '2024-09-15T10:00:00Z'
        }]
        
        response = client.get(f"/api/v1/patients/{patient_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['data']['demographics']['name'] == "Test Patient"
    
    @patch('app.routers.patients.require_patient_or_clinician_access')
    @patch('app.routers.patients.db_manager.execute_query')
    def test_get_patient_not_found(self, mock_db, mock_auth, mock_user):
        """Test patient retrieval when patient doesn't exist."""
        patient_id = str(uuid.uuid4())
        mock_auth.return_value = mock_user
        mock_db.return_value = []  # No results
        
        response = client.get(f"/api/v1/patients/{patient_id}")
        
        assert response.status_code == 404
    
    @patch('app.routers.patients.require_patient_or_clinician_access')
    @patch('app.routers.patients.db_manager.execute_query')
    def test_create_vital_success(self, mock_db, mock_auth, mock_user, sample_vital_data):
        """Test successful vital sign creation."""
        patient_id = str(uuid.uuid4())
        vital_id = str(uuid.uuid4())
        mock_auth.return_value = mock_user
        mock_db.return_value = [{
            'id': vital_id,
            'patient_id': patient_id,
            'type': sample_vital_data['type'],
            'value': sample_vital_data['value'],
            'unit': sample_vital_data['unit'],
            'source': sample_vital_data['source'],
            'ts': sample_vital_data['ts'],
            'created_at': '2024-09-15T10:00:00Z'
        }]
        
        vital_create_data = {**sample_vital_data, "patient_id": patient_id}
        response = client.post(f"/api/v1/patients/{patient_id}/vitals", json=vital_create_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert len(data['data']) == 1
        assert data['data'][0]['type'] == sample_vital_data['type']
    
    @patch('app.routers.patients.require_patient_or_clinician_access')
    @patch('app.routers.patients.db_manager.execute_query')
    def test_get_patient_vitals(self, mock_db, mock_auth, mock_user):
        """Test retrieving patient vitals."""
        patient_id = str(uuid.uuid4())
        mock_auth.return_value = mock_user
        mock_db.return_value = [
            {
                'type': 'glucose_fasting',
                'value': 95.0,
                'unit': 'mg/dL',
                'ts': '2024-09-15T08:00:00Z',
                'source': 'device'
            },
            {
                'type': 'sbp',
                'value': 120.0,
                'unit': 'mmHg',
                'ts': '2024-09-15T08:05:00Z',
                'source': 'device'
            }
        ]
        
        response = client.get(f"/api/v1/patients/{patient_id}/vitals")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert len(data['data']) == 2
    
    def test_invalid_patient_id_format(self):
        """Test API with invalid UUID format."""
        response = client.get("/api/v1/patients/invalid-uuid")
        
        assert response.status_code == 422  # Validation error


class TestPatientDataValidation:
    """Test patient data validation."""
    
    def test_demographics_validation(self):
        """Test demographics data validation."""
        invalid_data = {
            "demographics": {
                "age": -5,  # Invalid age
                "sex": "X",  # Invalid sex
                "bmi": 100   # Unrealistic BMI
            }
        }
        
        # This would be caught by Pydantic validation
        with pytest.raises(ValueError):
            from app.models.patients import PatientCreate
            PatientCreate(**invalid_data)
    
    def test_vital_validation(self):
        """Test vital signs validation."""
        from app.models.patients import VitalCreate
        
        # Test valid vital
        valid_vital = VitalCreate(
            patient_id=str(uuid.uuid4()),
            type="glucose_fasting",
            value=95.0,
            unit="mg/dL"
        )
        assert valid_vital.value == 95.0
        
        # Test invalid glucose value (would be caught by custom validator)
        with pytest.raises(ValueError):
            VitalCreate(
                patient_id=str(uuid.uuid4()),
                type="glucose_fasting",
                value=1000.0,  # Unrealistic glucose
                unit="mg/dL"
            )


@pytest.mark.asyncio
class TestPatientBusinessLogic:
    """Test patient-related business logic."""
    
    async def test_patient_summary_generation(self):
        """Test patient summary generation logic."""
        # This would test the actual business logic
        # that combines patient data, vitals, conditions, etc.
        pass
    
    async def test_risk_calculation_integration(self):
        """Test integration with risk calculation service."""
        # This would test the integration with ML service
        pass


class TestPatientSecurity:
    """Test patient data security and access control."""
    
    @patch('app.routers.patients.get_current_active_user')
    def test_patient_isolation(self, mock_auth):
        """Test that patients can only access their own data."""
        # Test would verify RLS and access control
        pass
    
    @patch('app.routers.patients.get_current_active_user')
    def test_clinician_access_control(self, mock_auth):
        """Test clinician access to assigned patients only."""
        # Test would verify clinician can only see assigned patients
        pass
    
    def test_audit_logging(self):
        """Test that patient access is properly audited."""
        # Test would verify audit trail creation
        pass


if __name__ == "__main__":
    pytest.main([__file__])
