

export interface RiskExplanation {
  factor: string;
  impact: string;
  magnitude: number;
  value: number | string;
}

export interface Recommendation {
  priority: 'low' | 'medium' | 'high' | 'critical';
  action: string;
  timeframe: string;
}

export interface TrendData {
  direction: 'increasing' | 'decreasing' | 'stable';
  magnitude: number;
}

export interface PatientTrends {
  glucose_trend?: TrendData;
  weight_trend?: TrendData;
  blood_pressure_trend?: TrendData;
  heart_rate_trend?: TrendData;
  [key: string]: TrendData | undefined;
}

export interface PatientPrediction {
  patient_id: string;
  risk_probability: number;
  risk_category: 'Low' | 'Medium' | 'High' | 'Critical';
  explanations: RiskExplanation[];
  recommendations: Recommendation[];
  trends: PatientTrends;
}

export interface PatientRiskSummary {
  patient_id: string;
  risk_probability: number;
  risk_category: 'Low' | 'Medium' | 'High' | 'Critical';
  top_risk_factor: string;
}

export interface RiskDistribution {
  low: number;
  medium: number;
  high: number;
  critical?: number;
}

export interface CohortSummaryStats {
  total_patients: number;
  high_risk_count: number;
  average_risk: number;
  risk_distribution: RiskDistribution;
}

export interface CohortSummary {
  summary_stats: CohortSummaryStats;
  patient_risks: PatientRiskSummary[];
}

export interface ModelMetrics {
  model_performance: {
    best_model: string;
    auc_roc: number;
    auprc: number;
    confusion_matrix: number[][];
  };
  feature_importance: Array<{
    feature: string;
    importance: number;
  }> | null;
  calibration?: {
    mean_predicted_value: number[];
    fraction_of_positives: number[];
  };
  clinical_interpretation: {
    model_reliability: string;
    recommended_use: string[];
    limitations: string[];
    clinical_value: string[];
  };
}

export interface PatientDemographics {
  patient_id: string;
  age: number;
  gender: 'M' | 'F' | 'Other';
  chronic_conditions: string[];
  medications: string[];
  last_visit: string;
}

export interface VitalSigns {
  timestamp: string;
  glucose: number;
  systolic_bp: number;
  diastolic_bp: number;
  heart_rate: number;
  weight: number;
  temperature?: number;
  oxygen_saturation?: number;
}

export interface PatientDetails {
  patient_id: string;
  demographics: PatientDemographics;
  vital_signs: VitalSigns[];
  risk_history: Array<{
    date: string;
    risk_score: number;
    risk_category: string;
  }>;
  recent_events: Array<{
    date: string;
    event_type: string;
    description: string;
    severity: 'low' | 'medium' | 'high';
  }>;
}

export interface HealthStatus {
  status: 'healthy' | 'warning' | 'error';
  message: string;
  timestamp: string;
  services: {
    database: boolean;
    ml_model: boolean;
    api: boolean;
  };
}


export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  error?: string;
}


export interface ChartDataPoint {
  name: string;
  value: number;
  color?: string;
}

export interface TimeSeriesDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}


export interface PatientFilters {
  riskCategory?: 'Low' | 'Medium' | 'High' | 'Critical' | 'All';
  searchTerm?: string;
  sortBy?: 'risk_probability' | 'patient_id' | 'last_updated';
  sortOrder?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
}


export interface ApiError {
  response?: {
    data?: {
      message?: string;
    };
    status?: number;
  };
  message?: string;
  code?: string;
}


export interface DashboardState {
  selectedPatient: string | null;
  activeView: 'cohort' | 'patient' | 'metrics';
  filters: PatientFilters;
  isLoading: boolean;
  error: string | null;
}
