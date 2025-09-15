import useSWR from 'swr';
import type {
  PatientPrediction,
  CohortSummary,
  ModelMetrics,
  PatientDetails,
  HealthStatus,
  PatientFilters,
  ApiResponse,
  ApiError
} from '@/types/healthcare';


const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';


const fetcher = async (url: string) => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};


export const API_ENDPOINTS = {
  health: `${API_BASE}/health`,
  predictPatient: (patientId: string) => `${API_BASE}/predict/${patientId}`,
  cohortSummary: `${API_BASE}/cohort_summary`,
  modelMetrics: `${API_BASE}/model_metrics`,
  listPatients: `${API_BASE}/patients`,
  patientDetails: (patientId: string) => `${API_BASE}/patient_details/${patientId}`,
} as const;


export const healthcareAPI = {
  
  getHealth: async (): Promise<HealthStatus> => {
    return fetcher(API_ENDPOINTS.health);
  },

  
  getPrediction: async (patientId: string): Promise<PatientPrediction> => {
    return fetcher(API_ENDPOINTS.predictPatient(patientId));
  },

  
  getCohortSummary: async (): Promise<CohortSummary> => {
    return fetcher(API_ENDPOINTS.cohortSummary);
  },

  
  getModelMetrics: async (): Promise<ModelMetrics> => {
    return fetcher(API_ENDPOINTS.modelMetrics);
  },

  
  getPatients: async (): Promise<string[]> => {
    return fetcher(API_ENDPOINTS.listPatients);
  },

  
  getPatientDetails: async (patientId: string): Promise<PatientDetails> => {
    return fetcher(API_ENDPOINTS.patientDetails(patientId));
  },
};


export const useHealthStatus = () => {
  const { data, error, isLoading, mutate } = useSWR<HealthStatus>(
    API_ENDPOINTS.health,
    fetcher,
    {
      refreshInterval: 30000, 
      revalidateOnFocus: true,
    }
  );

  return {
    health: data,
    isLoading,
    isError: error,
    refresh: mutate,
  };
};

export const usePatientPrediction = (patientId: string | null) => {
  const { data, error, isLoading, mutate } = useSWR<PatientPrediction>(
    patientId ? API_ENDPOINTS.predictPatient(patientId) : null,
    fetcher,
    {
      revalidateOnFocus: true,
      dedupingInterval: 60000, 
    }
  );

  return {
    prediction: data,
    isLoading,
    isError: error,
    refresh: mutate,
  };
};

export const useCohortSummary = () => {
  const { data, error, isLoading, mutate } = useSWR<CohortSummary>(
    API_ENDPOINTS.cohortSummary,
    fetcher,
    {
      refreshInterval: 300000, 
      revalidateOnFocus: true,
    }
  );

  return {
    cohort: data,
    isLoading,
    isError: error,
    refresh: mutate,
  };
};

export const useModelMetrics = () => {
  const { data, error, isLoading, mutate } = useSWR<ModelMetrics>(
    API_ENDPOINTS.modelMetrics,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 600000, 
    }
  );

  return {
    metrics: data,
    isLoading,
    isError: error,
    refresh: mutate,
  };
};

export const usePatientList = () => {
  const { data, error, isLoading, mutate } = useSWR<string[]>(
    API_ENDPOINTS.listPatients,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 300000, 
    }
  );

  return {
    patients: data || [],
    isLoading,
    isError: error,
    refresh: mutate,
  };
};

export const usePatientDetails = (patientId: string | null) => {
  const { data, error, isLoading, mutate } = useSWR<PatientDetails>(
    patientId ? API_ENDPOINTS.patientDetails(patientId) : null,
    fetcher,
    {
      revalidateOnFocus: true,
      dedupingInterval: 120000, 
    }
  );

  return {
    details: data,
    isLoading,
    isError: error,
    refresh: mutate,
  };
};


export const getRiskColor = (riskCategory: string): string => {
  switch (riskCategory.toLowerCase()) {
    case 'low':
      return 'text-green-600 bg-green-50 border-green-200';
    case 'medium':
      return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    case 'high':
      return 'text-orange-600 bg-orange-50 border-orange-200';
    case 'critical':
      return 'text-red-600 bg-red-50 border-red-200';
    default:
      return 'text-gray-600 bg-gray-50 border-gray-200';
  }
};

export const getRiskBadgeColor = (riskCategory: string): string => {
  switch (riskCategory.toLowerCase()) {
    case 'low':
      return 'bg-green-100 text-green-800';
    case 'medium':
      return 'bg-yellow-100 text-yellow-800';
    case 'high':
      return 'bg-orange-100 text-orange-800';
    case 'critical':
      return 'bg-red-100 text-red-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
};

export const formatRiskProbability = (probability: number): string => {
  return `${(probability * 100).toFixed(1)}%`;
};

export const getPriorityColor = (priority: string): string => {
  switch (priority.toLowerCase()) {
    case 'low':
      return 'text-blue-600';
    case 'medium':
      return 'text-yellow-600';
    case 'high':
      return 'text-orange-600';
    case 'critical':
      return 'text-red-600';
    default:
      return 'text-gray-600';
  }
};


export const handleApiError = (error: ApiError | Error | unknown): string => {
  
  if (typeof error === 'object' && error !== null && 'response' in error) {
    const resp = (error as any).response;
    if (resp?.data?.message) {
      return resp.data.message;
    }
  }

  
  if (error instanceof Error && error.message) {
    return error.message;
  }

  return 'An unexpected error occurred. Please try again.';
};

export const transformCohortDataForChart = (cohort: CohortSummary) => {
  const { risk_distribution } = cohort.summary_stats;
  return [
    { name: 'Low Risk', value: risk_distribution.low, color: '#10b981' },
    { name: 'Medium Risk', value: risk_distribution.medium, color: '#f59e0b' },
    { name: 'High Risk', value: risk_distribution.high, color: '#ef4444' },
    ...(risk_distribution.critical ? [{ name: 'Critical', value: risk_distribution.critical, color: '#dc2626' }] : [])
  ];
};

export const transformTrendsForChart = (trends: any) => {
  return Object.entries(trends).map(([key, trend]: [string, any]) => ({
    name: key.replace('_trend', '').replace('_', ' '),
    direction: trend.direction,
    magnitude: trend.magnitude,
    color: trend.direction === 'increasing' ? '#ef4444' : 
           trend.direction === 'decreasing' ? '#10b981' : '#6b7280'
  }));
};
