"""Evidence scoring and verification logic."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger()


class EvidenceScorer:
    """Score and verify evidence quality according to PRD guidelines."""
    
    def __init__(self):
        # Evidence type weights based on PRD rubric
        self.evidence_weights = {
            'guideline': 0.95,      # ADA guidelines - highest weight
            'meta_analysis': 0.90,  # Meta-analyses and systematic reviews
            'rct': 0.85,           # Randomized controlled trials
            'cohort': 0.70,        # Cohort studies
            'case_control': 0.60,  # Case-control studies
            'observational': 0.50,  # Observational studies
            'case_series': 0.30,   # Case series/reports
            'review': 0.40,        # Narrative reviews
            'expert_opinion': 0.25, # Expert opinions
            'other': 0.20          # Other evidence types
        }
        
        # Source quality modifiers
        self.source_modifiers = {
            'high_impact_journal': 0.1,   # High-impact journal bonus
            'recent_publication': 0.05,   # Recent publication bonus
            'large_sample': 0.05,         # Large sample size bonus
            'peer_reviewed': 0.02         # Peer review bonus
        }
        
        # Safety flags that require flagging
        self.safety_flags = [
            'black_box_warning',
            'contraindication',
            'severe_adverse_event',
            'drug_recall',
            'interaction_warning'
        ]
    
    def score_evidence_collection(self, evidence_collection: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Score a complete evidence collection."""
        try:
            if not evidence_collection:
                return {
                    'overall_score': 0.0,
                    'status': 'flagged',
                    'evidence_count': 0,
                    'quality_breakdown': {},
                    'flags': ['no_evidence_found'],
                    'warnings': []
                }
            
            # Score individual evidence items
            scored_evidence = []
            total_weighted_score = 0.0
            total_weight = 0.0
            quality_counts = {}
            all_flags = []
            all_warnings = []
            
            for evidence in evidence_collection:
                scored_item = self._score_individual_evidence(evidence)
                scored_evidence.append(scored_item)
                
                # Accumulate weighted scores
                weight = scored_item['weight']
                score = scored_item['quality_score']
                total_weighted_score += score * weight
                total_weight += weight
                
                # Count quality levels
                quality_level = scored_item['quality_level']
                quality_counts[quality_level] = quality_counts.get(quality_level, 0) + 1
                
                # Collect flags and warnings
                all_flags.extend(scored_item.get('flags', []))
                all_warnings.extend(scored_item.get('warnings', []))
            
            # Calculate overall score
            if total_weight > 0:
                overall_score = total_weighted_score / total_weight
            else:
                overall_score = 0.0
            
            # Apply collection-level adjustments
            overall_score = self._apply_collection_adjustments(
                overall_score, scored_evidence, quality_counts
            )
            
            # Determine status
            status = self._determine_evidence_status(overall_score, all_flags, all_warnings)
            
            result = {
                'overall_score': min(1.0, max(0.0, overall_score)),
                'status': status,
                'evidence_count': len(evidence_collection),
                'scored_evidence': scored_evidence,
                'quality_breakdown': quality_counts,
                'flags': list(set(all_flags)),  # Remove duplicates
                'warnings': list(set(all_warnings)),
                'scoring_metadata': {
                    'total_weight': total_weight,
                    'weighted_score': total_weighted_score,
                    'high_quality_evidence': quality_counts.get('high', 0),
                    'moderate_quality_evidence': quality_counts.get('moderate', 0),
                    'low_quality_evidence': quality_counts.get('low', 0)
                }
            }
            
            logger.info("Evidence collection scored", 
                       overall_score=overall_score,
                       status=status,
                       evidence_count=len(evidence_collection))
            
            return result
            
        except Exception as e:
            logger.error("Evidence scoring failed", error=str(e))
            return {
                'overall_score': 0.0,
                'status': 'flagged',
                'evidence_count': 0,
                'quality_breakdown': {},
                'flags': ['scoring_error'],
                'warnings': [f"Scoring failed: {str(e)}"]
            }
    
    def _score_individual_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Score an individual piece of evidence."""
        source_type = evidence.get('source_type', 'other')
        title = evidence.get('title', '')
        content = evidence.get('content', '') or evidence.get('snippet', '')
        metadata = evidence.get('metadata', {})
        
        # Base score from evidence type
        base_score = self.evidence_weights.get(source_type, self.evidence_weights['other'])
        
        # Apply quality modifiers
        quality_score = base_score
        applied_modifiers = []
        
        # Journal quality modifier
        if self._is_high_impact_source(evidence):
            quality_score += self.source_modifiers['high_impact_journal']
            applied_modifiers.append('high_impact_journal')
        
        # Recency modifier
        if self._is_recent_publication(evidence):
            quality_score += self.source_modifiers['recent_publication']
            applied_modifiers.append('recent_publication')
        
        # Sample size modifier
        if self._has_large_sample(evidence):
            quality_score += self.source_modifiers['large_sample']
            applied_modifiers.append('large_sample')
        
        # Study design quality
        study_quality_modifier = self._assess_study_design_quality(evidence)
        quality_score += study_quality_modifier
        
        # Cap the score at 1.0
        quality_score = min(1.0, quality_score)
        
        # Detect flags and warnings
        flags = self._detect_safety_flags(evidence)
        warnings = self._detect_warnings(evidence)
        
        # Determine quality level
        quality_level = self._categorize_quality_level(quality_score)
        
        # Calculate weight for this evidence
        weight = self._calculate_evidence_weight(source_type, quality_score, evidence)
        
        return {
            'source_type': source_type,
            'title': title,
            'quality_score': quality_score,
            'quality_level': quality_level,
            'weight': weight,
            'base_score': base_score,
            'modifiers_applied': applied_modifiers,
            'flags': flags,
            'warnings': warnings,
            'metadata': metadata,
            'relevance_score': self._assess_relevance(evidence)
        }
    
    def _is_high_impact_source(self, evidence: Dict[str, Any]) -> bool:
        """Check if evidence is from a high-impact source."""
        source_type = evidence.get('source_type', '')
        
        # Guidelines are always high impact
        if source_type == 'guideline':
            return True
        
        # Check journal impact
        journal = evidence.get('journal', '').lower()
        title = evidence.get('title', '').lower()
        
        high_impact_journals = [
            'new england journal of medicine', 'nejm',
            'lancet', 'jama', 'nature medicine',
            'diabetes care', 'circulation',
            'american journal of cardiology',
            'journal of the american college of cardiology',
            'european heart journal', 'bmj',
            'annals of internal medicine'
        ]
        
        return any(hij in journal for hij in high_impact_journals)
    
    def _is_recent_publication(self, evidence: Dict[str, Any]) -> bool:
        """Check if publication is recent (within 5 years)."""
        pub_date = evidence.get('pub_date', '') or evidence.get('publication_date', '')
        
        if not pub_date:
            return False
        
        try:
            # Try different date formats
            for date_format in ['%Y', '%Y/%m/%d', '%Y-%m-%d', '%Y %b']:
                try:
                    pub_dt = datetime.strptime(pub_date[:len(date_format)], date_format)
                    years_old = (datetime.now() - pub_dt).days / 365.25
                    return years_old <= 5
                except ValueError:
                    continue
        except Exception:
            pass
        
        return False
    
    def _has_large_sample(self, evidence: Dict[str, Any]) -> bool:
        """Check if study has large sample size."""
        metadata = evidence.get('metadata', {})
        sample_size = metadata.get('sample_size') or metadata.get('n_participants')
        
        if sample_size:
            try:
                n = int(sample_size)
                return n >= 1000  # Large sample threshold
            except (ValueError, TypeError):
                pass
        
        # Look for sample size in title or content
        content = (evidence.get('title', '') + ' ' + evidence.get('content', '')).lower()
        
        import re
        # Look for patterns like "n=1000", "1000 patients", etc.
        patterns = [
            r'n\s*=\s*(\d+)',
            r'(\d+)\s+patients?',
            r'(\d+)\s+participants?',
            r'(\d+)\s+subjects?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    n = int(match.replace(',', ''))
                    if n >= 1000:
                        return True
                except ValueError:
                    continue
        
        return False
    
    def _assess_study_design_quality(self, evidence: Dict[str, Any]) -> float:
        """Assess study design quality for additional scoring."""
        source_type = evidence.get('source_type', '')
        content = (evidence.get('title', '') + ' ' + evidence.get('content', '')).lower()
        
        quality_bonus = 0.0
        
        # RCT quality indicators
        if source_type == 'rct':
            if any(term in content for term in ['double-blind', 'double blind']):
                quality_bonus += 0.05
            if any(term in content for term in ['placebo-controlled', 'placebo controlled']):
                quality_bonus += 0.05
            if 'randomized' in content:
                quality_bonus += 0.02
        
        # Meta-analysis quality indicators
        elif source_type == 'meta_analysis':
            if 'systematic review' in content:
                quality_bonus += 0.05
            if any(term in content for term in ['cochrane', 'prisma']):
                quality_bonus += 0.03
        
        # Cohort study quality indicators
        elif source_type == 'cohort':
            if 'prospective' in content:
                quality_bonus += 0.03
            if any(term in content for term in ['follow-up', 'longitudinal']):
                quality_bonus += 0.02
        
        return quality_bonus
    
    def _detect_safety_flags(self, evidence: Dict[str, Any]) -> List[str]:
        """Detect safety flags that require attention."""
        flags = []
        content = (evidence.get('title', '') + ' ' + evidence.get('content', '')).lower()
        
        # Check for safety concerns
        safety_terms = {
            'black_box_warning': ['black box', 'boxed warning'],
            'contraindication': ['contraindicated', 'contraindication'],
            'severe_adverse_event': ['severe adverse', 'serious adverse', 'death', 'fatal'],
            'drug_recall': ['recall', 'withdrawn', 'discontinued'],
            'interaction_warning': ['drug interaction', 'contraindicated with']
        }
        
        for flag_type, terms in safety_terms.items():
            if any(term in content for term in terms):
                flags.append(flag_type)
        
        # Check metadata for safety flags
        metadata = evidence.get('metadata', {})
        if metadata.get('black_box_warning'):
            flags.append('black_box_warning')
        if metadata.get('safety_alert'):
            flags.append('safety_alert')
        
        return flags
    
    def _detect_warnings(self, evidence: Dict[str, Any]) -> List[str]:
        """Detect warnings that don't require flagging but need attention."""
        warnings = []
        content = (evidence.get('title', '') + ' ' + evidence.get('content', '')).lower()
        
        # Quality concerns
        if 'case report' in content or 'case series' in content:
            warnings.append('low_evidence_quality')
        
        if 'preliminary' in content or 'pilot study' in content:
            warnings.append('preliminary_evidence')
        
        if any(term in content for term in ['limited data', 'insufficient evidence']):
            warnings.append('limited_evidence')
        
        # Study limitations
        if any(term in content for term in ['small sample', 'limited sample']):
            warnings.append('small_sample_size')
        
        if 'retrospective' in content:
            warnings.append('retrospective_design')
        
        return warnings
    
    def _categorize_quality_level(self, quality_score: float) -> str:
        """Categorize quality level based on score."""
        if quality_score >= 0.8:
            return 'high'
        elif quality_score >= 0.6:
            return 'moderate'
        else:
            return 'low'
    
    def _calculate_evidence_weight(self, source_type: str, quality_score: float, 
                                 evidence: Dict[str, Any]) -> float:
        """Calculate weight for evidence in overall scoring."""
        # Base weight from source type
        base_weight = self.evidence_weights.get(source_type, 0.2)
        
        # Adjust by quality score
        weight = base_weight * quality_score
        
        # Relevance adjustment
        relevance = self._assess_relevance(evidence)
        weight *= relevance
        
        return weight
    
    def _assess_relevance(self, evidence: Dict[str, Any]) -> float:
        """Assess relevance of evidence to cardiometabolic care."""
        content = (evidence.get('title', '') + ' ' + evidence.get('content', '')).lower()
        
        # High relevance terms
        high_relevance_terms = [
            'diabetes', 'diabetic', 'glycemic', 'glucose', 'hba1c',
            'hypertension', 'blood pressure', 'cardiovascular',
            'cardiac', 'heart disease', 'cholesterol', 'lipid',
            'metabolic syndrome', 'obesity', 'bmi'
        ]
        
        # Moderate relevance terms
        moderate_relevance_terms = [
            'lifestyle', 'diet', 'exercise', 'weight loss',
            'medication', 'treatment', 'therapy', 'intervention'
        ]
        
        relevance_score = 0.5  # Base relevance
        
        # Count high relevance terms
        high_count = sum(1 for term in high_relevance_terms if term in content)
        relevance_score += min(0.4, high_count * 0.1)
        
        # Count moderate relevance terms
        moderate_count = sum(1 for term in moderate_relevance_terms if term in content)
        relevance_score += min(0.1, moderate_count * 0.02)
        
        return min(1.0, relevance_score)
    
    def _apply_collection_adjustments(self, base_score: float, 
                                    scored_evidence: List[Dict], 
                                    quality_counts: Dict[str, int]) -> float:
        """Apply collection-level scoring adjustments."""
        adjusted_score = base_score
        
        # Bonus for having multiple high-quality evidence pieces
        high_quality_count = quality_counts.get('high', 0)
        if high_quality_count >= 3:
            adjusted_score += 0.05
        elif high_quality_count >= 2:
            adjusted_score += 0.03
        
        # Bonus for evidence diversity
        source_types = set(e['source_type'] for e in scored_evidence)
        if len(source_types) >= 3:
            adjusted_score += 0.03
        elif len(source_types) >= 2:
            adjusted_score += 0.02
        
        # Penalty for too much low-quality evidence
        low_quality_count = quality_counts.get('low', 0)
        total_evidence = len(scored_evidence)
        if total_evidence > 0 and low_quality_count / total_evidence > 0.7:
            adjusted_score -= 0.05
        
        return adjusted_score
    
    def _determine_evidence_status(self, overall_score: float, 
                                 flags: List[str], warnings: List[str]) -> str:
        """Determine final evidence status."""
        # Check for blocking safety flags
        blocking_flags = [
            'black_box_warning', 'contraindication', 
            'severe_adverse_event', 'drug_recall'
        ]
        
        if any(flag in flags for flag in blocking_flags):
            return 'flagged'
        
        # Check score thresholds
        min_score = float(os.getenv('EVIDENCE_MIN_SCORE', '0.6'))
        
        if overall_score >= min_score:
            return 'approved'
        else:
            return 'flagged'


# Import os for environment variable
import os
