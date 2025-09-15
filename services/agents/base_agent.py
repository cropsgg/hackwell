"""Base agent class for MCP-style orchestration."""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger()


class AgentContext:
    """Context envelope for agent communication."""
    
    def __init__(self, patient_id: str, snapshot_ts: str = None):
        self.patient_id = patient_id
        self.snapshot_ts = snapshot_ts or datetime.utcnow().isoformat()
        self.patient_profile = {}
        self.vitals = []
        self.medications = []
        self.conditions = []
        self.agent_outputs = {}
        self.errors = []
        self.metadata = {}
    
    def add_patient_data(self, demographics: Dict[str, Any], vitals: List[Dict], 
                        medications: List[Dict], conditions: List[Dict]):
        """Add patient data to context."""
        self.patient_profile = demographics
        self.vitals = vitals
        self.medications = medications
        self.conditions = conditions
    
    def add_agent_output(self, agent_name: str, output: Dict[str, Any]):
        """Add output from an agent."""
        self.agent_outputs[agent_name] = output
    
    def get_agent_output(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get output from a specific agent."""
        return self.agent_outputs.get(agent_name)
    
    def add_error(self, agent_name: str, error: str):
        """Add error from an agent."""
        self.errors.append({
            'agent': agent_name,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            'patient_id': self.patient_id,
            'snapshot_ts': self.snapshot_ts,
            'patient_profile': self.patient_profile,
            'vitals': self.vitals,
            'medications': self.medications,
            'conditions': self.conditions,
            'agent_outputs': self.agent_outputs,
            'errors': self.errors,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentContext':
        """Create context from dictionary."""
        context = cls(data['patient_id'], data.get('snapshot_ts'))
        context.patient_profile = data.get('patient_profile', {})
        context.vitals = data.get('vitals', [])
        context.medications = data.get('medications', [])
        context.conditions = data.get('conditions', [])
        context.agent_outputs = data.get('agent_outputs', {})
        context.errors = data.get('errors', [])
        context.metadata = data.get('metadata', {})
        return context


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, description: str = None):
        self.name = name
        self.description = description or f"{name} agent"
        self.logger = structlog.get_logger().bind(agent=name)
    
    @abstractmethod
    async def process(self, context: AgentContext) -> AgentContext:
        """Process the context and return updated context."""
        pass
    
    def validate_context(self, context: AgentContext, 
                        required_fields: List[str] = None) -> bool:
        """Validate context has required fields."""
        if required_fields is None:
            required_fields = ['patient_id']
        
        for field in required_fields:
            if not hasattr(context, field) or getattr(context, field) is None:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        return True
    
    def handle_error(self, context: AgentContext, error: Exception) -> AgentContext:
        """Handle agent errors and update context."""
        error_msg = str(error)
        self.logger.error("Agent processing failed", error=error_msg)
        context.add_error(self.name, error_msg)
        return context
    
    def log_processing(self, context: AgentContext, action: str, **kwargs):
        """Log agent processing actions."""
        self.logger.info(
            f"Agent {action}",
            patient_id=context.patient_id,
            **kwargs
        )


class LLMAgent(BaseAgent):
    """Base class for LLM-powered agents."""
    
    def __init__(self, name: str, description: str = None, 
                 model: str = "gpt-4", temperature: float = 0.3):
        super().__init__(name, description)
        self.model = model
        self.temperature = temperature
        self.client = None  # Will be set by subclasses
    
    def build_prompt(self, context: AgentContext, template: str, **kwargs) -> str:
        """Build prompt from template and context."""
        try:
            from jinja2 import Template
            
            # Prepare template variables
            template_vars = {
                'patient_id': context.patient_id,
                'patient_profile': context.patient_profile,
                'vitals': context.vitals,
                'medications': context.medications,
                'conditions': context.conditions,
                'agent_outputs': context.agent_outputs,
                'snapshot_ts': context.snapshot_ts,
                **kwargs
            }
            
            template_obj = Template(template)
            return template_obj.render(**template_vars)
            
        except Exception as e:
            self.logger.error("Prompt building failed", error=str(e))
            return template  # Return raw template as fallback
    
    async def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call LLM with prompt."""
        # This is a placeholder - subclasses should implement specific LLM calls
        raise NotImplementedError("Subclasses must implement call_llm")
    
    def parse_llm_response(self, response: str, expected_format: str = "json") -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            if expected_format == "json":
                # Try to extract JSON from response
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                    return json.loads(json_str)
                else:
                    # Fallback: assume entire response is JSON
                    return json.loads(response)
            else:
                return {"text": response}
                
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse LLM response", error=str(e), response=response)
            return {"text": response, "parse_error": str(e)}
        except Exception as e:
            self.logger.error("Unexpected error parsing response", error=str(e))
            return {"text": response, "error": str(e)}


class OpenAIAgent(LLMAgent):
    """Agent using OpenAI API."""
    
    def __init__(self, name: str, description: str = None, 
                 model: str = "gpt-4", temperature: float = 0.3):
        super().__init__(name, description, model, temperature)
        self._setup_client()
    
    def _setup_client(self):
        """Setup OpenAI client."""
        try:
            import openai
            import os
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.logger.warning("OpenAI API key not found")
                return
            
            self.client = openai.OpenAI(api_key=api_key)
            
        except ImportError:
            self.logger.error("OpenAI library not installed")
        except Exception as e:
            self.logger.error("Failed to setup OpenAI client", error=str(e))
    
    async def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error("OpenAI API call failed", error=str(e))
            raise


class AnthropicAgent(LLMAgent):
    """Agent using Anthropic Claude API."""
    
    def __init__(self, name: str, description: str = None, 
                 model: str = "claude-3-sonnet-20240229", temperature: float = 0.3):
        super().__init__(name, description, model, temperature)
        self._setup_client()
    
    def _setup_client(self):
        """Setup Anthropic client."""
        try:
            import anthropic
            import os
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                self.logger.warning("Anthropic API key not found")
                return
            
            self.client = anthropic.Anthropic(api_key=api_key)
            
        except ImportError:
            self.logger.error("Anthropic library not installed")
        except Exception as e:
            self.logger.error("Failed to setup Anthropic client", error=str(e))
    
    async def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call Anthropic API."""
        if not self.client:
            raise RuntimeError("Anthropic client not available")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=self.temperature,
                system=system_prompt or "You are a helpful medical AI assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            self.logger.error("Anthropic API call failed", error=str(e))
            raise


class MockLLMAgent(LLMAgent):
    """Mock LLM agent for testing and development."""
    
    def __init__(self, name: str, description: str = None):
        super().__init__(name, description, "mock", 0.0)
    
    async def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Return mock response."""
        if "risk" in prompt.lower():
            return json.dumps({
                "interpretation": "Based on the patient data, there are several cardiovascular risk factors present.",
                "key_contributors": [
                    {"factor": "HbA1c", "impact": "high", "value": "8.1%"},
                    {"factor": "Blood pressure", "impact": "moderate", "value": "138/88"}
                ]
            })
        elif "care plan" in prompt.lower() or "recommendation" in prompt.lower():
            return json.dumps({
                "dietary": {
                    "recommendations": [
                        "Reduce refined carbohydrate intake",
                        "Increase fiber intake to 25-30g daily"
                    ]
                },
                "exercise": {
                    "aerobic": "150 minutes moderate-intensity exercise per week",
                    "resistance": "2-3 sessions per week"
                },
                "monitoring": {
                    "glucose": "Daily fasting glucose monitoring",
                    "blood_pressure": "Weekly home BP monitoring"
                }
            })
        else:
            return json.dumps({
                "status": "processed",
                "message": "Mock response generated",
                "data": {}
            })
