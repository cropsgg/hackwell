'use client';

import React, { useState, useMemo } from 'react';
import { Search, RefreshCw, Users, AlertTriangle, TrendingUp, Activity } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { useCohortSummary } from '@/services/api';
import { getRiskBadgeColor, formatRiskProbability, transformCohortDataForChart } from '@/services/api';
import type { PatientRiskSummary, PatientFilters } from '@/types/healthcare';

interface CohortViewProps {
  onPatientSelect: (patientId: string) => void;
  selectedPatient?: string | null;
}

export default function CohortView({ onPatientSelect, selectedPatient }: CohortViewProps) {
  const { cohort, isLoading, isError, refresh } = useCohortSummary();
  const [filters, setFilters] = useState<PatientFilters>({
    riskCategory: 'All',
    searchTerm: '',
    sortBy: 'risk_probability',
    sortOrder: 'desc',
  });

  
  const filteredPatients = useMemo(() => {
    if (!cohort?.patient_risks) return [];

    let filtered = cohort.patient_risks;

    
    if (filters.riskCategory && filters.riskCategory !== 'All') {
      filtered = filtered.filter((patient: PatientRiskSummary) => patient.risk_category === filters.riskCategory);
    }

    
    if (filters.searchTerm) {
      const searchLower = filters.searchTerm.toLowerCase();
      filtered = filtered.filter((patient: PatientRiskSummary) => 
        patient.patient_id.toLowerCase().includes(searchLower) ||
        patient.top_risk_factor.toLowerCase().includes(searchLower)
      );
    }

    
    filtered.sort((a: PatientRiskSummary, b: PatientRiskSummary) => {
      const aValue = a[filters.sortBy as keyof PatientRiskSummary];
      const bValue = b[filters.sortBy as keyof PatientRiskSummary];
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return filters.sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
      }
      
      const aStr = String(aValue).toLowerCase();
      const bStr = String(bValue).toLowerCase();
      return filters.sortOrder === 'asc' 
        ? aStr.localeCompare(bStr)
        : bStr.localeCompare(aStr);
    });

    return filtered;
  }, [cohort?.patient_risks, filters]);

  
  const riskDistributionData = cohort ? transformCohortDataForChart(cohort) : [];

  if (isError) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Failed to load cohort data. Please check your connection to the Flask backend.
          <Button variant="outline" size="sm" onClick={() => refresh()} className="ml-2">
            <RefreshCw className="h-4 w-4 mr-1" />
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Patient Cohort Overview</h1>
          <p className="text-muted-foreground">
            AI-driven risk assessment for chronic care patients
          </p>
        </div>
        <Button onClick={() => refresh()} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh Data
        </Button>
      </div>

      {/* Summary Statistics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Patients</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? <Skeleton className="h-8 w-16" /> : cohort?.summary_stats.total_patients.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">Active in system</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">High Risk Patients</CardTitle>
            <AlertTriangle className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">
              {isLoading ? <Skeleton className="h-8 w-16" /> : cohort?.summary_stats.high_risk_count.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              {cohort && `${((cohort.summary_stats.high_risk_count / cohort.summary_stats.total_patients) * 100).toFixed(1)}% of total`}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average Risk Score</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? <Skeleton className="h-8 w-16" /> : formatRiskProbability(cohort?.summary_stats.average_risk || 0)}
            </div>
            <p className="text-xs text-muted-foreground">Population average</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Monitoring</CardTitle>
            <Activity className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {isLoading ? <Skeleton className="h-8 w-16" /> : filteredPatients.length.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">Filtered results</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Risk Distribution Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Risk Distribution</CardTitle>
            <CardDescription>Patient population by risk category</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskDistributionData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {riskDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Risk Category Bar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Risk Category Breakdown</CardTitle>
            <CardDescription>Detailed view of patient risk levels</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={riskDistributionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8">
                    {riskDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Filters and Search */}
      <Card>
        <CardHeader>
          <CardTitle>Patient List</CardTitle>
          <CardDescription>Search and filter patients by risk level</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row gap-4 mb-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search patients or risk factors..."
                  value={filters.searchTerm}
                  onChange={(e) => setFilters(prev => ({ ...prev, searchTerm: e.target.value }))}
                  className="pl-8"
                />
              </div>
            </div>
            <Select
              value={filters.riskCategory}
              onValueChange={(value) => setFilters(prev => ({ ...prev, riskCategory: value as 'Low' | 'Medium' | 'High' | 'Critical' | 'All' }))}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Risk Category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="All">All Categories</SelectItem>
                <SelectItem value="Low">Low Risk</SelectItem>
                <SelectItem value="Medium">Medium Risk</SelectItem>
                <SelectItem value="High">High Risk</SelectItem>
                <SelectItem value="Critical">Critical Risk</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={filters.sortBy}
              onValueChange={(value) => setFilters(prev => ({ ...prev, sortBy: value as 'risk_probability' | 'patient_id' | 'last_updated' }))}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Sort By" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="risk_probability">Risk Score</SelectItem>
                <SelectItem value="patient_id">Patient ID</SelectItem>
                <SelectItem value="risk_category">Risk Category</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Patient Table */}
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Patient ID</TableHead>
                  <TableHead>Risk Score</TableHead>
                  <TableHead>Risk Category</TableHead>
                  <TableHead>Top Risk Factor</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <TableRow key={i}>
                      <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                      <TableCell><Skeleton className="h-4 w-16" /></TableCell>
                      <TableCell><Skeleton className="h-6 w-20" /></TableCell>
                      <TableCell><Skeleton className="h-4 w-32" /></TableCell>
                      <TableCell><Skeleton className="h-8 w-16" /></TableCell>
                    </TableRow>
                  ))
                ) : (
                  filteredPatients.map((patient: PatientRiskSummary) => (
                    <TableRow 
                      key={patient.patient_id}
                      className={selectedPatient === patient.patient_id ? 'bg-muted' : ''}
                    >
                      <TableCell className="font-medium">{patient.patient_id}</TableCell>
                      <TableCell>{formatRiskProbability(patient.risk_probability)}</TableCell>
                      <TableCell>
                        <Badge className={getRiskBadgeColor(patient.risk_category)}>
                          {patient.risk_category}
                        </Badge>
                      </TableCell>
                      <TableCell className="max-w-xs truncate">{patient.top_risk_factor}</TableCell>
                      <TableCell>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => onPatientSelect(patient.patient_id)}
                        >
                          View Details
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {filteredPatients.length === 0 && !isLoading && (
            <div className="text-center py-8 text-muted-foreground">
              No patients found matching the current filters.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
