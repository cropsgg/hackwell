'use client';

import React from 'react';
import { ArrowLeft, AlertTriangle, TrendingUp, TrendingDown, Minus, Calendar, User, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { usePatientPrediction, usePatientDetails } from '@/services/api';
import { getRiskBadgeColor, formatRiskProbability, getPriorityColor } from '@/services/api';
import type { RiskExplanation, Recommendation, TrendData } from '@/types/healthcare';

interface PatientDetailProps {
  patientId: string;
  onBack: () => void;
}

export default function PatientDetail({ patientId, onBack }: PatientDetailProps) {
  const { prediction, isLoading: predictionLoading, isError: predictionError } = usePatientPrediction(patientId);
  const { details, isLoading: detailsLoading, isError: detailsError } = usePatientDetails(patientId);

  const isLoading = predictionLoading || detailsLoading;
  const hasError = predictionError || detailsError;

  
  const vitalSignsData = details?.vital_signs?.map(vs => ({
    date: new Date(vs.timestamp).toLocaleDateString(),
    glucose: vs.glucose,
    systolic: vs.systolic_bp,
    diastolic: vs.diastolic_bp,
    heartRate: vs.heart_rate,
    weight: vs.weight,
  })) || [];

  
  const riskHistoryData = details?.risk_history?.map(rh => ({
    date: new Date(rh.date).toLocaleDateString(),
    riskScore: rh.risk_score * 100,
    category: rh.risk_category,
  })) || [];

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'increasing':
        return <TrendingUp className="h-4 w-4 text-red-500" />;
      case 'decreasing':
        return <TrendingDown className="h-4 w-4 text-green-500" />;
      default:
        return <Minus className="h-4 w-4 text-gray-500" />;
    }
  };

  const getRiskGaugeColor = (probability: number) => {
    if (probability >= 0.7) return 'text-red-600';
    if (probability >= 0.4) return 'text-orange-600';
    if (probability >= 0.2) return 'text-yellow-600';
    return 'text-green-600';
  };

  if (hasError) {
    return (
      <div className="p-6">
        <Button variant="outline" onClick={onBack} className="mb-4">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Cohort
        </Button>
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Failed to load patient data. Please check your connection and try again.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Button variant="outline" onClick={onBack}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Cohort
          </Button>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              {isLoading ? <Skeleton className="h-8 w-32" /> : `Patient ${patientId}`}
            </h1>
            <p className="text-muted-foreground">Detailed risk assessment and clinical insights</p>
          </div>
        </div>
        {prediction && (
          <Badge className={`text-lg px-4 py-2 ${getRiskBadgeColor(prediction.risk_category)}`}>
            {prediction.risk_category} Risk
          </Badge>
        )}
      </div>

      {/* Risk Score Gauge */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Current Risk Assessment</span>
          </CardTitle>
          <CardDescription>90-day deterioration risk probability</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-32 w-full" />
          ) : prediction ? (
            <div className="space-y-4">
              <div className="text-center">
                <div className={`text-6xl font-bold ${getRiskGaugeColor(prediction.risk_probability)}`}>
                  {formatRiskProbability(prediction.risk_probability)}
                </div>
                <p className="text-muted-foreground mt-2">Risk Probability</p>
              </div>
              <Progress 
                value={prediction.risk_probability * 100} 
                className="h-4"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Low Risk</span>
                <span>Medium Risk</span>
                <span>High Risk</span>
                <span>Critical</span>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="recommendations">Actions</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Risk Explanations */}
            <Card>
              <CardHeader>
                <CardTitle>Risk Factors</CardTitle>
                <CardDescription>Key factors contributing to patient risk</CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="space-y-3">
                    {Array.from({ length: 4 }).map((_, i) => (
                      <Skeleton key={i} className="h-16 w-full" />
                    ))}
                  </div>
                ) : prediction?.explanations ? (
                  <div className="space-y-4">
                    {prediction.explanations.map((explanation: RiskExplanation, index: number) => (
                      <div key={index} className="border rounded-lg p-4">
                        <div className="flex justify-between items-start mb-2">
                          <h4 className="font-medium">{explanation.factor}</h4>
                          <Badge variant="outline">
                            Impact: {(explanation.magnitude * 100).toFixed(0)}%
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mb-2">
                          {explanation.impact}
                        </p>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm">Current Value:</span>
                          <span className="font-medium">{explanation.value}</span>
                        </div>
                        <Progress value={explanation.magnitude * 100} className="mt-2 h-2" />
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-muted-foreground">No risk factors available</p>
                )}
              </CardContent>
            </Card>

            {/* Patient Demographics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <User className="h-5 w-5" />
                  <span>Patient Information</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="space-y-3">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                ) : details?.demographics ? (
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Age:</span>
                      <span>{details.demographics.age} years</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Gender:</span>
                      <span>{details.demographics.gender}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Last Visit:</span>
                      <span>{new Date(details.demographics.last_visit).toLocaleDateString()}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Conditions:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {details.demographics.chronic_conditions.map((condition: string, index: number) => (
                          <Badge key={index} variant="secondary" className="text-xs">
                            {condition}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Medications:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {details.demographics.medications.map((medication: string, index: number) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {medication}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-muted-foreground">No patient information available</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Trends Tab */}
        <TabsContent value="trends" className="space-y-6">
          <div className="grid gap-6">
            {/* Vital Signs Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Vital Signs Trends</CardTitle>
                <CardDescription>Historical vital signs over time</CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <Skeleton className="h-64 w-full" />
                ) : vitalSignsData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={vitalSignsData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="glucose" stroke="#8884d8" name="Glucose" />
                      <Line type="monotone" dataKey="systolic" stroke="#82ca9d" name="Systolic BP" />
                      <Line type="monotone" dataKey="heartRate" stroke="#ffc658" name="Heart Rate" />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="text-muted-foreground text-center py-8">No vital signs data available</p>
                )}
              </CardContent>
            </Card>

            {/* Trend Indicators */}
            <Card>
              <CardHeader>
                <CardTitle>Current Trends</CardTitle>
                <CardDescription>Direction and magnitude of key health indicators</CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                    {Array.from({ length: 4 }).map((_, i) => (
                      <Skeleton key={i} className="h-20 w-full" />
                    ))}
                  </div>
                ) : prediction?.trends ? (
                  <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                    {Object.entries(prediction.trends).map(([key, trend]) => {
                      if (!trend) return null;
                      const trendData = trend as TrendData;
                      return (
                        <div key={key} className="border rounded-lg p-4 text-center">
                          <div className="flex items-center justify-center mb-2">
                            {getTrendIcon(trendData.direction)}
                          </div>
                          <h4 className="font-medium capitalize">
                            {key.replace('_trend', '').replace('_', ' ')}
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            {trendData.direction} ({trendData.magnitude.toFixed(1)})
                          </p>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-muted-foreground">No trend data available</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Risk Score History</CardTitle>
              <CardDescription>Historical risk assessment over time</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Skeleton className="h-64 w-full" />
              ) : riskHistoryData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={riskHistoryData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value) => [`${value}%`, 'Risk Score']} />
                    <Area 
                      type="monotone" 
                      dataKey="riskScore" 
                      stroke="#8884d8" 
                      fill="#8884d8" 
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-muted-foreground text-center py-8">No risk history available</p>
              )}
            </CardContent>
          </Card>

          {/* Recent Events */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Calendar className="h-5 w-5" />
                <span>Recent Events</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : details?.recent_events?.length ? (
                <div className="space-y-3">
                  {details.recent_events.map((event, index: number) => (
                    <div key={index} className="border rounded-lg p-4">
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="font-medium">{event.event_type}</h4>
                        <Badge className={getPriorityColor(event.severity)}>
                          {event.severity}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-1">
                        {event.description}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(event.date).toLocaleDateString()}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground">No recent events recorded</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Recommendations Tab */}
        <TabsContent value="recommendations" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Clinical Recommendations</CardTitle>
              <CardDescription>Suggested actions based on current risk assessment</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <Skeleton key={i} className="h-20 w-full" />
                  ))}
                </div>
              ) : prediction?.recommendations?.length ? (
                <div className="space-y-4">
                  {prediction.recommendations.map((rec: Recommendation, index: number) => (
                    <div key={index} className="border rounded-lg p-4">
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="font-medium">{rec.action}</h4>
                        <Badge className={getPriorityColor(rec.priority)}>
                          {rec.priority} Priority
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Timeframe: {rec.timeframe}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground">No recommendations available</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
