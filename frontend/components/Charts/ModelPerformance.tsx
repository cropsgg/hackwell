'use client';

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, Cell } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertTriangle, TrendingUp, Target, BarChart3 } from 'lucide-react';
import { useModelMetrics } from '@/services/api';
import type { ModelMetrics } from '@/types/healthcare';

export default function ModelPerformance() {
  const { metrics, isLoading, isError, refresh } = useModelMetrics();

  
  const transformConfusionMatrix = (matrix: number[][]) => {
    const data = [];
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        data.push({
          predicted: j,
          actual: i,
          value: matrix[i][j],
          label: `Actual: ${i}, Predicted: ${j}`,
        });
      }
    }
    return data;
  };

  
  const getConfusionMatrixColor = (value: number, max: number) => {
    const intensity = value / max;
    return `rgba(59, 130, 246, ${intensity})`;
  };

  if (isError) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Failed to load model performance metrics. Please check your connection to the Flask backend.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Model Performance</h1>
        <p className="text-muted-foreground">
          AI model metrics and performance analysis
        </p>
      </div>

      {/* Performance Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">AUC-ROC</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? <Skeleton className="h-8 w-16" /> : (metrics?.model_performance?.auc_roc || 0).toFixed(3)}
            </div>
            <p className="text-xs text-muted-foreground">Area under ROC curve</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">AU-PRC</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? <Skeleton className="h-8 w-16" /> : (metrics?.model_performance?.auprc || 0).toFixed(3)}
            </div>
            <p className="text-xs text-muted-foreground">Area under precision-recall curve</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Best Model</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-lg font-bold">
              {isLoading ? <Skeleton className="h-8 w-24" /> : (metrics?.model_performance?.best_model || 'Unknown')}
            </div>
            <p className="text-xs text-muted-foreground">Selected algorithm</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Feature Importance */}
        <Card>
          <CardHeader>
            <CardTitle>Feature Importance</CardTitle>
            <CardDescription>Most influential factors in risk prediction</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : metrics?.feature_importance ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={metrics.feature_importance.slice(0, 10)} 
                  layout="horizontal"
                  margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={80} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#8884d8">
                    {metrics.feature_importance.slice(0, 10).map((entry: { feature: string; importance: number }, index: number) => (
                      <Cell key={`cell-${index}`} fill={`hsl(${220 + index * 20}, 70%, 50%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-muted-foreground text-center py-8">No feature importance data available</p>
            )}
          </CardContent>
        </Card>

        {/* ROC Curve */}
        <Card>
          <CardHeader>
            <CardTitle>ROC Curve</CardTitle>
            <CardDescription>Receiver Operating Characteristic</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : metrics?.model_performance?.roc_curve ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={metrics.model_performance.roc_curve.fpr.map((fpr: number, index: number) => ({
                    fpr,
                    tpr: metrics.model_performance.roc_curve!.tpr[index],
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="fpr" 
                    domain={[0, 1]}
                    label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    domain={[0, 1]}
                    label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    formatter={(value, name) => [
                      typeof value === 'number' ? value.toFixed(3) : value, 
                      name === 'tpr' ? 'True Positive Rate' : 'False Positive Rate'
                    ]}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="tpr" 
                    stroke="#8884d8" 
                    strokeWidth={2}
                    dot={false}
                  />
                  {/* Diagonal reference line */}
                  <Line 
                    type="monotone" 
                    dataKey="fpr" 
                    stroke="#cccccc" 
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-muted-foreground text-center py-8">No ROC curve data available</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Confusion Matrix */}
      <Card>
        <CardHeader>
          <CardTitle>Confusion Matrix</CardTitle>
          <CardDescription>Model prediction accuracy breakdown</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-64 w-full" />
          ) : metrics?.model_performance?.confusion_matrix ? (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-2 max-w-md mx-auto">
                <div></div>
                <div className="text-center font-medium text-sm">Predicted</div>
                <div></div>
                <div className="flex items-center justify-center font-medium text-sm">
                  <span className="transform -rotate-90">Actual</span>
                </div>
                <div className="grid grid-cols-2 gap-1">
                  {metrics.model_performance.confusion_matrix.map((row: number[], i: number) =>
                    row.map((value: number, j: number) => {
                      const maxValue = Math.max(...metrics.model_performance.confusion_matrix!.flat());
                      const isCorrect = i === j;
                      return (
                        <div
                          key={`${i}-${j}`}
                          className={`
                            h-16 w-16 flex items-center justify-center text-white font-bold rounded
                            ${isCorrect ? 'bg-green-500' : 'bg-red-500'}
                          `}
                          style={{
                            opacity: 0.3 + (value / maxValue) * 0.7
                          }}
                        >
                          {value}
                        </div>
                      );
                    })
                  )}
                </div>
                <div></div>
              </div>
              <div className="flex justify-center space-x-4 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-green-500 rounded"></div>
                  <span>Correct Predictions</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-red-500 rounded"></div>
                  <span>Incorrect Predictions</span>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground text-center py-8">No confusion matrix data available</p>
          )}
        </CardContent>
      </Card>

      {/* Calibration Plot */}
      {metrics?.calibration && (
        <Card>
          <CardHeader>
            <CardTitle>Calibration Plot</CardTitle>
            <CardDescription>Model probability calibration assessment</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart
                data={metrics.calibration.mean_predicted_value.map((pred: number, index: number) => ({
                  predicted: pred,
                  observed: metrics.calibration!.fraction_of_positives[index],
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="predicted" 
                  domain={[0, 1]}
                  label={{ value: 'Mean Predicted Probability', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  domain={[0, 1]}
                  label={{ value: 'Fraction of Positives', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value) => [typeof value === 'number' ? value.toFixed(3) : value]}
                />
                <Scatter dataKey="observed" fill="#8884d8" />
                {/* Perfect calibration line */}
                <Line 
                  type="monotone" 
                  dataKey="predicted" 
                  stroke="#cccccc" 
                  strokeDasharray="5 5"
                  dot={false}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Model Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Model Summary</CardTitle>
          <CardDescription>Key performance indicators and model health</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ) : metrics ? (
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <h4 className="font-medium mb-2">Performance Metrics</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>AUC-ROC:</span>
                    <Badge variant="outline">{(metrics.model_performance?.auc_roc ?? 0).toFixed(3)}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>AU-PRC:</span>
                    <Badge variant="outline">{(metrics.model_performance?.auprc ?? 0).toFixed(3)}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Best Model:</span>
                    <Badge variant="outline">{metrics.model_performance?.best_model ?? 'Unknown'}</Badge>
                  </div>
                </div>
              </div>
              <div>
                <h4 className="font-medium mb-2">Model Status</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Model Health:</span>
                    <Badge className={(metrics.model_performance?.auc_roc ?? 0) > 0.8 ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}>
                      {(metrics.model_performance?.auc_roc ?? 0) > 0.8 ? 'Excellent' : 'Good'}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Features Used:</span>
                    <Badge variant="outline">{metrics.feature_importance?.length ?? 0}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Last Updated:</span>
                    <span className="text-muted-foreground">Recently</span>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
