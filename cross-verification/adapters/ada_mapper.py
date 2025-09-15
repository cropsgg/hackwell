"""ADA Standards of Care mapper for curated guideline statements."""

import yaml
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class ADAStatement(BaseModel):
    """ADA Standards of Care statement."""
    id: str
    title: str
    content: str
    section: str
    subsection: Optional[str] = None
    evidence_level: str  # A, B, C, E
    recommendation_class: str  # 1, 2, 3
    conditions: List[str] = []
    keywords: List[str] = []
    citations: List[str] = []
    url: Optional[str] = None
    last_updated: Optional[str] = None


class ADAMapper:
    """Mapper for ADA Standards of Care statements."""
    
    def __init__(self, guidelines_file: Optional[str] = None):
        self.guidelines_file = guidelines_file
        self.statements: List[ADAStatement] = []
        self._load_guidelines()
    
    def _load_guidelines(self):
        """Load ADA guidelines from file or use default curated statements."""
        if self.guidelines_file:
            try:
                with open(self.guidelines_file, 'r') as f:
                    data = yaml.safe_load(f)
                    self.statements = [ADAStatement(**stmt) for stmt in data.get('statements', [])]
                logger.info("ADA guidelines loaded from file", 
                           file=self.guidelines_file,
                           statements=len(self.statements))
            except Exception as e:
                logger.error("Failed to load ADA guidelines file", 
                           file=self.guidelines_file,
                           error=str(e))
                self._load_default_guidelines()
        else:
            self._load_default_guidelines()
    
    def _load_default_guidelines(self):
        """Load default curated ADA statements."""
        self.statements = [
            ADAStatement(
                id="ada_exercise_001",
                title="Physical Activity for Type 2 Diabetes",
                content="Adults with type 2 diabetes should engage in at least 150 minutes of moderate-intensity aerobic activity per week, spread over at least 3 days per week, with no more than 2 consecutive days without activity.",
                section="Physical Activity",
                subsection="Aerobic Exercise",
                evidence_level="A",
                recommendation_class="1",
                conditions=["type 2 diabetes"],
                keywords=["exercise", "aerobic", "physical activity", "150 minutes", "moderate intensity"],
                citations=["American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Supplement_1):S1-S291."],
                url="https://diabetesjournals.org/care/article/46/Supplement_1/S1/148027/Standards-of-Medical-Care-in-Diabetes-2023",
                last_updated="2023-01-01"
            ),
            ADAStatement(
                id="ada_exercise_002",
                title="Resistance Training for Diabetes",
                content="Adults with type 2 diabetes should engage in resistance training at least 2-3 times per week on nonconsecutive days.",
                section="Physical Activity",
                subsection="Resistance Training",
                evidence_level="A",
                recommendation_class="1",
                conditions=["type 2 diabetes"],
                keywords=["resistance training", "strength training", "2-3 times per week"],
                citations=["American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Supplement_1):S1-S291."],
                url="https://diabetesjournals.org/care/article/46/Supplement_1/S1/148027/Standards-of-Medical-Care-in-Diabetes-2023",
                last_updated="2023-01-01"
            ),
            ADAStatement(
                id="ada_metformin_001",
                title="Metformin as First-Line Therapy",
                content="Metformin is the preferred initial pharmacologic agent for the treatment of type 2 diabetes. It should be initiated at the time of diabetes diagnosis along with lifestyle modification.",
                section="Pharmacologic Approaches to Glycemic Treatment",
                subsection="Metformin",
                evidence_level="A",
                recommendation_class="1",
                conditions=["type 2 diabetes"],
                keywords=["metformin", "first-line", "initial therapy", "pharmacologic treatment"],
                citations=["American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Supplement_1):S1-S291."],
                url="https://diabetesjournals.org/care/article/46/Supplement_1/S1/148027/Standards-of-Medical-Care-in-Diabetes-2023",
                last_updated="2023-01-01"
            ),
            ADAStatement(
                id="ada_metformin_002",
                title="Metformin Contraindications",
                content="Metformin should not be used in patients with eGFR <30 mL/min/1.73 m² and should be used with caution in patients with eGFR 30-45 mL/min/1.73 m².",
                section="Pharmacologic Approaches to Glycemic Treatment",
                subsection="Metformin Safety",
                evidence_level="A",
                recommendation_class="1",
                conditions=["type 2 diabetes", "chronic kidney disease"],
                keywords=["metformin", "contraindication", "eGFR", "kidney disease", "safety"],
                citations=["American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Supplement_1):S1-S291."],
                url="https://diabetesjournals.org/care/article/46/Supplement_1/S1/148027/Standards-of-Medical-Care-in-Diabetes-2023",
                last_updated="2023-01-01"
            ),
            ADAStatement(
                id="ada_sulfonylurea_001",
                title="Sulfonylurea Hypoglycemia Risk",
                content="Sulfonylureas are associated with an increased risk of hypoglycemia, particularly in elderly patients and those with renal impairment. They should be used with caution in these populations.",
                section="Pharmacologic Approaches to Glycemic Treatment",
                subsection="Sulfonylureas",
                evidence_level="A",
                recommendation_class="1",
                conditions=["type 2 diabetes", "elderly", "chronic kidney disease"],
                keywords=["sulfonylurea", "hypoglycemia", "elderly", "renal impairment", "safety"],
                citations=["American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Supplement_1):S1-S291."],
                url="https://diabetesjournals.org/care/article/46/Supplement_1/S1/148027/Standards-of-Medical-Care-in-Diabetes-2023",
                last_updated="2023-01-01"
            ),
            ADAStatement(
                id="ada_diet_001",
                title="Carbohydrate Management",
                content="For people with diabetes, there is no single ideal percentage of calories from carbohydrate, protein, and fat. The macronutrient distribution should be individualized based on current eating patterns, preferences, and metabolic goals.",
                section="Nutrition Therapy",
                subsection="Carbohydrate Management",
                evidence_level="B",
                recommendation_class="1",
                conditions=["type 1 diabetes", "type 2 diabetes"],
                keywords=["carbohydrate", "macronutrients", "nutrition", "individualized"],
                citations=["American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Supplement_1):S1-S291."],
                url="https://diabetesjournals.org/care/article/46/Supplement_1/S1/148027/Standards-of-Medical-Care-in-Diabetes-2023",
                last_updated="2023-01-01"
            ),
            ADAStatement(
                id="ada_diet_002",
                title="Fiber Intake",
                content="People with diabetes should consume at least 14 g fiber per 1,000 kcal or at least 25-35 g fiber per day from a variety of food sources.",
                section="Nutrition Therapy",
                subsection="Fiber",
                evidence_level="B",
                recommendation_class="1",
                conditions=["type 1 diabetes", "type 2 diabetes"],
                keywords=["fiber", "nutrition", "25-35 g", "dietary fiber"],
                citations=["American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Supplement_1):S1-S291."],
                url="https://diabetesjournals.org/care/article/46/Supplement_1/S1/148027/Standards-of-Medical-Care-in-Diabetes-2023",
                last_updated="2023-01-01"
            ),
            ADAStatement(
                id="ada_monitoring_001",
                title="HbA1c Monitoring Frequency",
                content="For patients meeting treatment goals, assess HbA1c at least twice annually. For patients not meeting treatment goals or with therapy changes, assess HbA1c quarterly.",
                section="Assessment of Glycemic Control",
                subsection="HbA1c Testing",
                evidence_level="A",
                recommendation_class="1",
                conditions=["type 1 diabetes", "type 2 diabetes"],
                keywords=["HbA1c", "monitoring", "quarterly", "twice annually"],
                citations=["American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Supplement_1):S1-S291."],
                url="https://diabetesjournals.org/care/article/46/Supplement_1/S1/148027/Standards-of-Medical-Care-in-Diabetes-2023",
                last_updated="2023-01-01"
            ),
            ADAStatement(
                id="ada_glp1_001",
                title="GLP-1 Receptor Agonists for Cardiovascular Risk",
                content="In patients with type 2 diabetes and established cardiovascular disease, a GLP-1 receptor agonist with proven cardiovascular benefit is recommended to reduce major adverse cardiovascular events.",
                section="Cardiovascular Disease and Risk Management",
                subsection="GLP-1 Receptor Agonists",
                evidence_level="A",
                recommendation_class="1",
                conditions=["type 2 diabetes", "cardiovascular disease"],
                keywords=["GLP-1", "cardiovascular", "CVD", "MACE"],
                citations=["American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Supplement_1):S1-S291."],
                url="https://diabetesjournals.org/care/article/46/Supplement_1/S1/148027/Standards-of-Medical-Care-in-Diabetes-2023",
                last_updated="2023-01-01"
            ),
            ADAStatement(
                id="ada_sglt2_001",
                title="SGLT2 Inhibitors for Heart Failure",
                content="In patients with type 2 diabetes and heart failure, an SGLT2 inhibitor with proven benefit in this population is recommended to reduce heart failure hospitalization and cardiovascular mortality.",
                section="Cardiovascular Disease and Risk Management",
                subsection="SGLT2 Inhibitors",
                evidence_level="A",
                recommendation_class="1",
                conditions=["type 2 diabetes", "heart failure"],
                keywords=["SGLT2", "heart failure", "cardiovascular mortality"],
                citations=["American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Supplement_1):S1-S291."],
                url="https://diabetesjournals.org/care/article/46/Supplement_1/S1/148027/Standards-of-Medical-Care-in-Diabetes-2023",
                last_updated="2023-01-01"
            )
        ]
        
        logger.info("Default ADA guidelines loaded", statements=len(self.statements))
    
    def search_statements(
        self, 
        query: str, 
        conditions: Optional[List[str]] = None,
        evidence_levels: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[ADAStatement]:
        """Search ADA statements by query and filters."""
        try:
            query_lower = query.lower()
            matching_statements = []
            
            for statement in self.statements:
                # Check if query matches content or keywords
                content_match = query_lower in statement.content.lower()
                title_match = query_lower in statement.title.lower()
                keyword_match = any(query_lower in keyword.lower() for keyword in statement.keywords)
                
                if content_match or title_match or keyword_match:
                    # Apply condition filter
                    if conditions:
                        if not any(cond.lower() in [c.lower() for c in statement.conditions] for cond in conditions):
                            continue
                    
                    # Apply evidence level filter
                    if evidence_levels:
                        if statement.evidence_level not in evidence_levels:
                            continue
                    
                    matching_statements.append(statement)
            
            # Sort by evidence level and recommendation class
            matching_statements.sort(
                key=lambda x: (x.evidence_level, x.recommendation_class),
                reverse=True
            )
            
            logger.info("ADA statement search completed",
                       query=query,
                       matches=len(matching_statements),
                       max_results=max_results)
            
            return matching_statements[:max_results]
            
        except Exception as e:
            logger.error("ADA statement search failed", query=query, error=str(e))
            return []
    
    def get_statements_by_category(
        self, 
        category: str, 
        max_results: int = 10
    ) -> List[ADAStatement]:
        """Get ADA statements by category (exercise, medication, diet, etc.)."""
        try:
            category_lower = category.lower()
            matching_statements = []
            
            for statement in self.statements:
                # Check section and subsection
                section_match = category_lower in statement.section.lower()
                subsection_match = statement.subsection and category_lower in statement.subsection.lower()
                keyword_match = any(category_lower in keyword.lower() for keyword in statement.keywords)
                
                if section_match or subsection_match or keyword_match:
                    matching_statements.append(statement)
            
            # Sort by evidence level
            matching_statements.sort(key=lambda x: x.evidence_level, reverse=True)
            
            return matching_statements[:max_results]
            
        except Exception as e:
            logger.error("Failed to get statements by category", 
                        category=category, 
                        error=str(e))
            return []
    
    def get_statement_by_id(self, statement_id: str) -> Optional[ADAStatement]:
        """Get a specific ADA statement by ID."""
        for statement in self.statements:
            if statement.id == statement_id:
                return statement
        return None
    
    def get_all_statements(self) -> List[ADAStatement]:
        """Get all ADA statements."""
        return self.statements.copy()
    
    def add_statement(self, statement: ADAStatement):
        """Add a new ADA statement."""
        self.statements.append(statement)
        logger.info("ADA statement added", statement_id=statement.id)
    
    def update_statement(self, statement_id: str, updates: Dict[str, Any]):
        """Update an existing ADA statement."""
        for i, statement in enumerate(self.statements):
            if statement.id == statement_id:
                for key, value in updates.items():
                    if hasattr(statement, key):
                        setattr(statement, key, value)
                logger.info("ADA statement updated", statement_id=statement_id)
                return True
        return False
    
    def remove_statement(self, statement_id: str) -> bool:
        """Remove an ADA statement."""
        for i, statement in enumerate(self.statements):
            if statement.id == statement_id:
                del self.statements[i]
                logger.info("ADA statement removed", statement_id=statement_id)
                return True
        return False


def create_ada_mapper(guidelines_file: Optional[str] = None) -> ADAMapper:
    """Factory function to create ADA mapper."""
    return ADAMapper(guidelines_file=guidelines_file)
