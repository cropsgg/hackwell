'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import RiskBadgeComponent from '@/components/Dashboard/RiskBadge';
import EvidenceCard from '@/components/Dashboard/EvidenceCard';
import RecommendationCard from '@/components/Dashboard/RecommendationCard';
import TimelineChart from '@/components/Dashboard/TimelineChart';
import TopBar from '@/components/Dashboard/TopBar';
import { 
  Heart, 
  Activity, 
  Droplets, 
  Brain, 
  AlertCircle, 
  Clock, 
  User, 
  FileText, 
  MessageCircle, 
  Menu,
  BookOpen,
  Pill,
  Stethoscope,
  Send,
  Home,
  Target,
  Users,
  TrendingUp,
  CheckCircle,
  X,
  ExternalLink,
  Filter,
  Search,
  BarChart3,
  LineChart,
  Shield
} from 'lucide-react';

// Mock data for clinician dashboard - Patient List
const mockPatients = [
  {
    id: "SJ-2024-001",
    name: "Sarah Johnson",
    age: 45,
    mrn: "MRN-789456",
    riskLevel: "High" as const,
    riskScore: 85,
    priority: "urgent" as const,
    lastVisit: "2024-03-15",
    nextAppointment: "2024-03-22",
    conditions: ["Type 2 Diabetes", "Hypertension"],
    vitals: { bp: "158/95", glucose: 180, hr: 88, weight: 168 },
    alerts: 2,
    status: "needs_review" as const
  },
  {
    id: "MR-2024-002", 
    name: "Michael Rodriguez",
    age: 52,
    mrn: "MRN-456123",
    riskLevel: "Medium" as const,
    riskScore: 68,
    priority: "moderate" as const,
    lastVisit: "2024-03-14",
    nextAppointment: "2024-03-28",
    conditions: ["Heart Disease", "Hypertension"],
    vitals: { bp: "142/89", glucose: 145, hr: 75, weight: 185 },
    alerts: 1,
    status: "stable" as const
  },
  {
    id: "EW-2024-003",
    name: "Emma Wilson", 
    age: 38,
    mrn: "MRN-321789",
    riskLevel: "Low" as const,
    riskScore: 32,
    priority: "routine" as const,
    lastVisit: "2024-03-10",
    nextAppointment: "2024-04-10",
    conditions: ["Pre-diabetes"],
    vitals: { bp: "125/82", glucose: 110, hr: 72, weight: 145 },
    alerts: 0,
    status: "stable" as const
  },
  {
    id: "JD-2024-004",
    name: "James Davis",
    age: 61,
    mrn: "MRN-654987",
    riskLevel: "High" as const,
    riskScore: 92,
    priority: "urgent" as const,
    lastVisit: "2024-03-16",
    nextAppointment: "2024-03-20",
    conditions: ["Type 2 Diabetes", "Heart Disease", "Hypertension"],
    vitals: { bp: "165/98", glucose: 220, hr: 95, weight: 195 },
    alerts: 3,
    status: "critical" as const
  }
];

const selectedPatient = mockPatients[0]; // Sarah Johnson for detailed view

const mockClinician = {
  name: "Dr. Michael Chen",
  role: "Primary Care Physician",
  department: "Internal Medicine"
};

const mockEvidence = [
  {
    title: "ADA Guidelines for Diabetes Management",
    description: "Current evidence supports HbA1c targets <7% for most adults with diabetes to reduce microvascular complications.",
    evidenceLevel: "A" as const,
    source: "American Diabetes Association, 2024",
    type: "guideline" as const,
    url: "#"
  },
  {
    title: "Framingham Heart Study Risk Assessment",
    description: "10-year cardiovascular risk calculation based on age, gender, cholesterol levels, and blood pressure readings.",
    evidenceLevel: "A" as const,
    source: "Framingham Heart Study, 2023",
    type: "study" as const,
    url: "#"
  },
  {
    title: "DASH Diet for Hypertension Control",
    description: "Dietary approaches to stop hypertension show significant reduction in systolic BP (8-14 mmHg reduction).",
    evidenceLevel: "B" as const,
    source: "NEJM Hypertension Research, 2023",
    type: "research" as const,
    url: "#"
  }
];

const mockRecommendations = [
  {
    title: "Reduce Sodium Intake",
    description: "Limit daily sodium to <2,300mg. Focus on fresh foods and read nutrition labels carefully.",
    category: "diet" as const,
    priority: "high" as const,
    timeframe: "Start immediately",
    completed: false
  },
  {
    title: "30-Minute Daily Walking",
    description: "Moderate aerobic exercise 5 days per week to improve cardiovascular health and glucose control.",
    category: "exercise" as const,
    priority: "high" as const,
    timeframe: "Begin this week",
    completed: false
  },
  {
    title: "Metformin Adherence Check",
    description: "Ensure consistent timing and dosage. Schedule medication review with pharmacist if needed.",
    category: "medication" as const,
    priority: "medium" as const,
    timeframe: "Next 2 weeks",
    completed: true
  },
  {
    title: "Blood Pressure Monitoring",
    description: "Daily home BP monitoring for 2 weeks, then 3x weekly. Log readings in health app.",
    category: "lifestyle" as const,
    priority: "high" as const,
    timeframe: "Start tomorrow",
    completed: false
  }
];

const mockTimelineEvents = [
  {
    date: "March 15, 2024",
    type: "assessment" as const,
    title: "Quarterly Health Assessment",
    description: "Comprehensive metabolic panel, HbA1c, and cardiovascular risk evaluation completed.",
    value: 7.2,
    trend: "stable" as const
  },
  {
    date: "February 28, 2024",
    type: "intervention" as const,
    title: "Nutrition Counseling Session",
    description: "Met with registered dietitian to develop personalized meal plan for diabetes management."
  },
  {
    date: "February 10, 2024",
    type: "alert" as const,
    title: "Elevated Blood Pressure Reading",
    description: "Home monitoring showed consistent readings >140/90. Medication adjustment recommended.",
    value: 148,
    trend: "up" as const
  },
  {
    date: "January 20, 2024",
    type: "milestone" as const,
    title: "Weight Loss Goal Achieved",
    description: "Successfully lost 10 pounds through dietary changes and increased physical activity.",
    value: 165,
    trend: "down" as const
  }
];

export default function ClinicianDashboard() {
  const [activeTab, setActiveTab] = useState('patients');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedPatientId, setSelectedPatientId] = useState(mockPatients[0].id);
  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState('all');

  const currentPatient = mockPatients.find(p => p.id === selectedPatientId) || mockPatients[0];

  const sidebarItems = [
    { id: 'patients', label: 'Patient List', icon: <Users className="w-5 h-5" /> },
    { id: 'timeline', label: 'Patient Timeline', icon: <LineChart className="w-5 h-5" /> },
    { id: 'explainability', label: 'Model Insights', icon: <Brain className="w-5 h-5" /> },
    { id: 'evidence', label: 'Evidence Panel', icon: <BookOpen className="w-5 h-5" /> },
    { id: 'recommendations', label: 'Recommendations', icon: <Target className="w-5 h-5" /> },
    { id: 'audit', label: 'Audit Trail', icon: <FileText className="w-5 h-5" /> },
  ];

  return (
    <div className="min-h-screen bg-white">
      {/* Top Bar */}
      <TopBar 
        currentPatient={currentPatient}
        clinicianName={mockClinician.name}
        onPatientSelect={() => console.log('Patient selector clicked')}
        onNotifications={() => console.log('Notifications clicked')}
        onSettings={() => console.log('Settings clicked')}
      />
      
      <div className="flex">
        {/* Sidebar */}
        <div className={`${sidebarCollapsed ? 'w-16' : 'w-64'} bg-white border-r border-gray-200 transition-all duration-300 flex flex-col min-h-[calc(100vh-73px)]`}>
          {/* Sidebar Header */}
          <div className="p-6 border-b border-gray-100">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="w-full justify-start text-gray-600 hover:text-gray-900 hover:bg-gray-50"
            >
              <Menu className="w-5 h-5" />
              {!sidebarCollapsed && <span className="ml-3 font-medium">Menu</span>}
            </Button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4">
            <div className="space-y-1">
              {sidebarItems.map((item) => (
                <Button
                  key={item.id}
                  variant={activeTab === item.id ? "default" : "ghost"}
                  className={`w-full justify-start transition-all duration-200 ${
                    activeTab === item.id 
                      ? 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-sm' 
                      : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                  }`}
                  onClick={() => setActiveTab(item.id)}
                >
                  {item.icon}
                  {!sidebarCollapsed && <span className="ml-3 font-medium">{item.label}</span>}
                </Button>
              ))}
            </div>
          </nav>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-auto">
          <div className="p-8">
            {activeTab === 'patients' && (
              <div className="space-y-6">
                {/* Patient List Header */}
                <div className="flex justify-between items-center">
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">Patient Dashboard</h1>
                    <p className="text-gray-600">Manage patient triage and risk stratification</p>
                  </div>
                  <div className="flex gap-3">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                      <Input 
                        placeholder="Search patients..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="pl-10 w-64"
                      />
                    </div>
                    <select 
                      value={riskFilter}
                      onChange={(e) => setRiskFilter(e.target.value)}
                      className="px-3 py-2 border rounded-md text-sm"
                    >
                      <option value="all">All Risk Levels</option>
                      <option value="High">High Risk</option>
                      <option value="Medium">Medium Risk</option>
                      <option value="Low">Low Risk</option>
                    </select>
                  </div>
                </div>

                {/* Patient List */}
                <Card className="border-0 shadow-sm">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-lg font-medium text-gray-900">Patient List</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="space-y-2">
                      {mockPatients.map((patient) => (
                        <div 
                          key={patient.id}
                          className={`p-4 rounded-lg cursor-pointer transition-all hover:bg-gray-50 ${
                            selectedPatientId === patient.id ? 'bg-green-50 border border-green-200' : 'bg-white border border-gray-100'
                          }`}
                          onClick={() => setSelectedPatientId(patient.id)}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <div className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">
                                <User className="w-5 h-5 text-gray-500" />
                              </div>
                              <div>
                                <h3 className="font-medium text-gray-900">{patient.name}</h3>
                                <p className="text-sm text-gray-500">Age {patient.age} • {patient.conditions[0]}</p>
                              </div>
                            </div>
                            <div className="flex items-center gap-3">
                              <div className="text-right">
                                <Badge 
                                  variant={patient.riskLevel === 'High' ? 'destructive' : 'outline'}
                                  className={patient.riskLevel === 'High' ? 'text-white' : 'text-green-700 border-green-200 bg-green-50'}
                                >
                                  {patient.riskLevel}
                                </Badge>
                                <p className="text-xs text-gray-400 mt-1">{patient.lastVisit}</p>
                              </div>
                              {patient.alerts > 0 && (
                                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'timeline' && (
              <div className="space-y-6">
                {/* Selected Patient Header */}
                <div className="bg-white border border-gray-100 rounded-lg p-6">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center">
                      <User className="w-6 h-6 text-gray-500" />
                    </div>
                    <div className="flex-1">
                      <h1 className="text-xl font-semibold text-gray-900">{currentPatient.name}</h1>
                      <p className="text-sm text-gray-500">Age {currentPatient.age} • {currentPatient.conditions[0]}</p>
                    </div>
                    <div className="text-right">
                      <Badge 
                        variant={currentPatient.riskLevel === 'High' ? 'destructive' : 'outline'}
                        className={currentPatient.riskLevel === 'High' ? 'text-white' : 'text-green-700 border-green-200 bg-green-50'}
                      >
                        {currentPatient.riskLevel} Risk
                      </Badge>
                      <p className="text-xs text-gray-400 mt-1">Score: {currentPatient.riskScore}</p>
                    </div>
                  </div>
                </div>

                {/* Vitals Overview */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white border border-gray-100 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-600">Blood Pressure</span>
                      <Heart className="h-4 w-4 text-green-600" />
                    </div>
                    <div className="text-lg font-semibold text-gray-900">{currentPatient.vitals.bp}</div>
                  </div>

                  <div className="bg-white border border-gray-100 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-600">Glucose</span>
                      <Droplets className="h-4 w-4 text-green-600" />
                    </div>
                    <div className="text-lg font-semibold text-gray-900">{currentPatient.vitals.glucose}</div>
                  </div>

                  <div className="bg-white border border-gray-100 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-600">Heart Rate</span>
                      <Activity className="h-4 w-4 text-green-600" />
                    </div>
                    <div className="text-lg font-semibold text-gray-900">{currentPatient.vitals.hr}</div>
                  </div>

                  <div className="bg-white border border-gray-100 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-600">Weight</span>
                      <User className="h-4 w-4 text-green-600" />
                    </div>
                    <div className="text-lg font-semibold text-gray-900">{currentPatient.vitals.weight}</div>
                  </div>
                </div>

                {/* Timeline Chart */}
                <Card className="rounded-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5" />
                      Patient Timeline & History
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <TimelineChart events={mockTimelineEvents} />
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'explainability' && (
              <div className="space-y-6">
                <div className="bg-white border border-gray-100 rounded-lg p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">Model Insights</h2>
                      <p className="text-sm text-gray-500">AI decision factors for {currentPatient.name}</p>
                    </div>
                    <Badge variant="outline" className="text-green-700 border-green-200 bg-green-50">
                      87% Confidence
                    </Badge>
                  </div>
                </div>

                {/* Key Risk Factors */}
                <div className="bg-white border border-gray-100 rounded-lg p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Key Risk Factors</h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                        <span className="font-medium text-gray-900">Blood Pressure</span>
                        <span className="text-sm text-gray-500">158/95 mmHg</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-24 bg-gray-200 rounded-full h-1.5">
                          <div className="bg-red-500 h-1.5 rounded-full" style={{width: '85%'}}></div>
                        </div>
                        <span className="text-sm font-medium text-gray-700">High Impact</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                        <span className="font-medium text-gray-900">Glucose Level</span>
                        <span className="text-sm text-gray-500">180 mg/dL</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-24 bg-gray-200 rounded-full h-1.5">
                          <div className="bg-orange-500 h-1.5 rounded-full" style={{width: '70%'}}></div>
                        </div>
                        <span className="text-sm font-medium text-gray-700">Medium Impact</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="font-medium text-gray-900">Exercise Frequency</span>
                        <span className="text-sm text-gray-500">Regular</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-24 bg-gray-200 rounded-full h-1.5">
                          <div className="bg-green-500 h-1.5 rounded-full" style={{width: '30%'}}></div>
                        </div>
                        <span className="text-sm font-medium text-gray-700">Protective</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Clinical Summary */}
                <div className="bg-white border border-gray-100 rounded-lg p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Clinical Summary</h3>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-gray-700 leading-relaxed">
                      Patient shows elevated cardiovascular risk due to uncontrolled hypertension (158/95 mmHg) 
                      and hyperglycemia (180 mg/dL). Regular exercise provides protective benefit. 
                      Priority focus on blood pressure and glucose management recommended.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'evidence' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">Evidence Panel</h2>
                    <p className="text-gray-600">Clinical guidelines and research for {currentPatient.name}</p>
                  </div>
                  <Button variant="outline" size="sm">
                    <ExternalLink className="w-4 h-4 mr-2" />
                    Search PubMed
                  </Button>
                </div>

                {/* Evidence Cards with External Links */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="rounded-lg">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">ADA Diabetes Guidelines</CardTitle>
                        <Badge className="bg-green-100 text-green-800">Level A</Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-600 mb-4">
                        Standards of Medical Care in Diabetes—2024. Glycemic targets for adults with diabetes.
                      </p>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="text-xs">
                          <ExternalLink className="w-3 h-3 mr-1" />
                          ADA Guidelines
                        </Button>
                        <Button variant="outline" size="sm" className="text-xs">
                          <ExternalLink className="w-3 h-3 mr-1" />
                          PubMed: 38265592
                        </Button>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="rounded-lg">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">ACC/AHA Hypertension Guidelines</CardTitle>
                        <Badge className="bg-green-100 text-green-800">Level A</Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-600 mb-4">
                        2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults.
                      </p>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="text-xs">
                          <ExternalLink className="w-3 h-3 mr-1" />
                          ACC/AHA Guidelines
                        </Button>
                        <Button variant="outline" size="sm" className="text-xs">
                          <ExternalLink className="w-3 h-3 mr-1" />
                          PubMed: 29146535
                        </Button>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="rounded-lg">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">FDA Drug Safety Communication</CardTitle>
                        <Badge className="bg-blue-100 text-blue-800">FDA Alert</Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-600 mb-4">
                        SGLT2 inhibitors: Drug Safety Communication - increased risk of necrotizing fasciitis of the perineum.
                      </p>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="text-xs">
                          <ExternalLink className="w-3 h-3 mr-1" />
                          openFDA Database
                        </Button>
                        <Button variant="outline" size="sm" className="text-xs">
                          <ExternalLink className="w-3 h-3 mr-1" />
                          FDA Safety Alert
                        </Button>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="rounded-lg">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">Cochrane Systematic Review</CardTitle>
                        <Badge className="bg-green-100 text-green-800">Level A</Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-600 mb-4">
                        Blood pressure targets for the treatment of people with hypertension and cardiovascular disease.
                      </p>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="text-xs">
                          <ExternalLink className="w-3 h-3 mr-1" />
                          Cochrane Library
                        </Button>
                        <Button variant="outline" size="sm" className="text-xs">
                          <ExternalLink className="w-3 h-3 mr-1" />
                          PubMed: 35138684
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}

            {activeTab === 'recommendations' && (
              <div className="space-y-6">
                <div className="bg-white border border-gray-100 rounded-lg p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">Treatment Recommendations</h2>
                      <p className="text-sm text-gray-500">Clinical recommendations for {currentPatient.name}</p>
                    </div>
                    <Badge variant="outline" className="text-green-700 border-green-200 bg-green-50">
                      3 Pending
                    </Badge>
                  </div>
                </div>

                {/* Recommendation Cards */}
                <div className="space-y-3">
                  {mockRecommendations.map((recommendation, index) => (
                    <div key={index} className="bg-white border border-gray-100 rounded-lg p-6">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-3">
                            <Badge 
                              variant={recommendation.priority === 'high' ? 'destructive' : 'outline'}
                              className={recommendation.priority === 'high' ? 'text-white' : 'text-green-700 border-green-200 bg-green-50'}
                            >
                              {recommendation.priority === 'high' ? 'High Priority' : 'Standard'}
                            </Badge>
                            <span className="text-xs text-gray-400">{recommendation.category}</span>
                          </div>
                          <h3 className="text-lg font-medium text-gray-900 mb-2">{recommendation.title}</h3>
                          <p className="text-gray-600 mb-3">{recommendation.description}</p>
                          <p className="text-sm text-gray-500">
                            {recommendation.timeframe}
                          </p>
                        </div>
                        <div className="flex gap-2 ml-6">
                          {!recommendation.completed ? (
                            <>
                              <Button size="sm" className="bg-green-600 hover:bg-green-700 text-white">
                                <CheckCircle className="w-4 h-4 mr-1" />
                                Approve
                              </Button>
                              <Button variant="outline" size="sm" className="text-gray-600 border-gray-300">
                                Review
                              </Button>
                            </>
                          ) : (
                            <div className="flex items-center gap-1 text-green-600 text-sm">
                              <CheckCircle className="w-4 h-4" />
                              Approved
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'audit' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">Audit Trail</h2>
                    <p className="text-gray-600">System activity and compliance tracking</p>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <Filter className="w-4 h-4 mr-2" />
                      Filter
                    </Button>
                    <Button variant="outline" size="sm">
                      <FileText className="w-4 h-4 mr-2" />
                      Export
                    </Button>
                  </div>
                </div>

                {/* Audit Trail Entries */}
                <Card className="rounded-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="w-5 h-5" />
                      Recent Activity
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {[
                        { 
                          time: "2024-03-15 14:30", 
                          user: "Dr. Michael Chen", 
                          action: "Approved AI recommendation: Blood Pressure Monitoring", 
                          type: "approval",
                          details: "Approved with modifications - frequency changed to daily"
                        },
                        { 
                          time: "2024-03-15 14:25", 
                          user: "System", 
                          action: "Updated risk scores based on new lab results", 
                          type: "update",
                          details: "HbA1c: 7.2% → Risk score increased from 82 to 85"
                        },
                        { 
                          time: "2024-03-15 14:20", 
                          user: "Dr. Michael Chen", 
                          action: "Overrode AI recommendation: Metformin dosage increase", 
                          type: "override",
                          details: "Reason: Patient reported GI intolerance at higher doses"
                        },
                        { 
                          time: "2024-03-15 14:15", 
                          user: "Sarah Johnson", 
                          action: "Patient portal login - vitals updated", 
                          type: "login",
                          details: "BP: 158/95, Weight: 168 lbs, Glucose: 180 mg/dL"
                        },
                        { 
                          time: "2024-03-15 14:10", 
                          user: "System", 
                          action: "Automated risk assessment completed", 
                          type: "system",
                          details: "Model confidence: 87% - 4 risk factors identified"
                        }
                      ].map((entry, index) => (
                        <div key={index} className="border border-gray-200 rounded-lg p-4">
                          <div className="flex items-start justify-between">
                            <div className="flex items-start gap-3">
                              <div className={`w-3 h-3 rounded-full mt-1 ${
                                entry.type === 'approval' ? 'bg-green-500' :
                                entry.type === 'override' ? 'bg-red-500' :
                                entry.type === 'update' ? 'bg-amber-500' :
                                entry.type === 'login' ? 'bg-purple-500' :
                                'bg-gray-500'
                              }`}></div>
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                  <p className="text-sm font-medium text-gray-900">{entry.action}</p>
                                  <Badge 
                                    variant={entry.type === 'approval' ? 'default' : 
                                            entry.type === 'override' ? 'destructive' : 'secondary'}
                                    className="text-xs"
                                  >
                                    {entry.type.toUpperCase()}
                                  </Badge>
                                </div>
                                <p className="text-xs text-gray-500 mb-2">{entry.user} • {entry.time}</p>
                                <p className="text-xs text-gray-600 bg-gray-50 p-2 rounded">{entry.details}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
