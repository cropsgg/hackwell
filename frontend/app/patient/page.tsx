'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
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
  Calendar,
  CheckCircle,
  Target,
  TrendingUp,
  Home,
  Shield,
  Bell,
  Plus,
  Settings,
  BarChart3,
  Send,
  Moon,
  AlertTriangle,
  Award,
  Stethoscope,
  Dumbbell,
  Apple,
  Zap
} from 'lucide-react';

// Mock patient data for wellness-focused portal
const mockPatient = {
  id: "SJ-2024-001",
  name: "Sarah Johnson",
  firstName: "Sarah",
  age: 45,
  gender: "Female",
  avatar: "/api/placeholder/64/64",
  conditions: ["Type 2 Diabetes", "Hypertension"],
  riskLevel: "Medium" as const,
  riskScore: 72,
  lastUpdate: "2024-03-15",
  onboardingComplete: false,
  vitals: {
    bloodPressure: "142/89",
    heartRate: 78,
    bloodSugar: 156,
    weight: 165,
    steps: 8420,
    sleep: 7.2,
    lastBP: "2024-03-15 18:30",
    lastGlucose: "2024-03-15 14:20"
  },
  goals: {
    dailySteps: 10000,
    weeklyExercise: 150, // minutes
    weightTarget: 155,
    bpTarget: "130/80"
  },
  streaks: {
    medication: 12,
    exercise: 5,
    vitalsLogging: 8
  }
};

// Daily wellness plan with personalized suggestions
const mockDailyPlan = [
  {
    id: 1,
    title: "Morning Medication",
    description: "Metformin 500mg with breakfast - helps control blood sugar",
    category: "medication",
    icon: <Pill className="w-5 h-5" />,
    completed: true,
    time: "8:00 AM",
    streak: 12,
    importance: "critical"
  },
  {
    id: 2,
    title: "Log Blood Glucose",
    description: "Check and record your morning glucose level",
    category: "monitoring",
    icon: <Droplets className="w-5 h-5" />,
    completed: true,
    time: "Before breakfast",
    streak: 8,
    importance: "high"
  },
  {
    id: 3,
    title: "30-Minute Walk",
    description: "Light exercise after lunch - helps lower blood sugar naturally",
    category: "exercise",
    icon: <Activity className="w-5 h-5" />,
    completed: false,
    time: "After lunch",
    streak: 5,
    importance: "high"
  },
  {
    id: 4,
    title: "Afternoon BP Check",
    description: "Record your blood pressure reading",
    category: "monitoring",
    icon: <Heart className="w-5 h-5" />,
    completed: false,
    time: "3:00 PM",
    streak: 8,
    importance: "medium"
  },
  {
    id: 5,
    title: "Healthy Dinner Prep",
    description: "Low-sodium meal with lean protein and vegetables",
    category: "nutrition",
    icon: <Apple className="w-5 h-5" />,
    completed: false,
    time: "6:30 PM",
    streak: 0,
    importance: "medium"
  },
  {
    id: 6,
    title: "Evening Medication",
    description: "Lisinopril 10mg for blood pressure management",
    category: "medication",
    icon: <Pill className="w-5 h-5" />,
    completed: false,
    time: "8:00 PM",
    streak: 12,
    importance: "critical"
  }
];

// Medication reminders with interaction warnings
const mockMedications = [
  {
    id: 1,
    name: "Metformin",
    dosage: "500mg",
    frequency: "Twice daily",
    nextDose: "8:00 PM",
    instructions: "Take with food to reduce stomach upset",
    sideEffects: ["Nausea", "Diarrhea", "Metallic taste"],
    interactions: ["Avoid alcohol", "Monitor with contrast dyes"]
  },
  {
    id: 2,
    name: "Lisinopril",
    dosage: "10mg",
    frequency: "Once daily",
    nextDose: "8:00 PM",
    instructions: "Take at the same time each day",
    sideEffects: ["Dry cough", "Dizziness", "Fatigue"],
    interactions: ["Monitor potassium levels", "Avoid NSAIDs"]
  }
];

const mockReminders = [
  {
    id: 1,
    title: "Evening Medication",
    message: "Don't forget your blood pressure medication",
    time: "6:00 PM",
    type: "medication",
    urgent: false
  },
  {
    id: 2,
    title: "Doctor Appointment",
    message: "Quarterly check-up with Dr. Chen tomorrow",
    time: "Tomorrow 10:00 AM",
    type: "appointment",
    urgent: true
  },
  {
    id: 3,
    title: "Hydration Reminder",
    message: "You're doing great! Try to drink more water today",
    time: "Now",
    type: "wellness",
    urgent: false
  }
];

const mockTimelineData = [
  { date: "Mon", glucose: 145, bp: 138, steps: 8420 },
  { date: "Tue", glucose: 152, bp: 142, steps: 7890 },
  { date: "Wed", glucose: 148, bp: 140, steps: 9200 },
  { date: "Thu", glucose: 156, bp: 139, steps: 8650 },
  { date: "Fri", glucose: 151, bp: 141, steps: 9100 },
  { date: "Sat", glucose: 149, bp: 138, steps: 10200 },
  { date: "Today", glucose: 156, bp: 142, steps: 8420 }
];

export default function PatientDashboard() {
  const [activeTab, setActiveTab] = useState('welcome');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [completedTasks, setCompletedTasks] = useState(new Set([1]));
  const [chatMessage, setChatMessage] = useState('');

  const toggleTask = (taskId: number) => {
    const newCompleted = new Set(completedTasks);
    if (newCompleted.has(taskId)) {
      newCompleted.delete(taskId);
    } else {
      newCompleted.add(taskId);
    }
    setCompletedTasks(newCompleted);
  };

  const completedCount = completedTasks.size;
  const progressPercentage = (completedCount / mockDailyPlan.length) * 100;

  const tabs = [
    { id: 'home', label: 'Home', icon: <Home className="w-4 h-4" /> },
    { id: 'onboarding', label: 'Getting Started', icon: <Stethoscope className="w-4 h-4" /> },
    { id: 'daily-plan', label: 'Daily Plan', icon: <CheckCircle className="w-4 h-4" /> },
    { id: 'vitals', label: 'Log Vitals', icon: <Heart className="w-4 h-4" /> },
    { id: 'medications', label: 'Medications', icon: <Pill className="w-4 h-4" /> },
    { id: 'progress', label: 'My Progress', icon: <TrendingUp className="w-4 h-4" /> },
    { id: 'coach', label: 'Health Coach', icon: <MessageCircle className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <div className="bg-white border-b border-gray-100 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-green-600 rounded-lg flex items-center justify-center">
                <Heart className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">CareSight</h1>
                <p className="text-sm text-green-600">Your Health Companion</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Bell className="w-5 h-5 text-gray-400 hover:text-green-600 cursor-pointer" />
              <div className="w-8 h-8 bg-green-50 rounded-full flex items-center justify-center">
                <User className="w-5 h-5 text-green-600" />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex">
        {/* Sidebar */}
        <div className={`${sidebarCollapsed ? 'w-16' : 'w-64'} bg-white border-r border-gray-200 transition-all duration-300 flex flex-col min-h-[calc(100vh-65px)]`}>
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

          {/* Navigation Tabs */}
          {/* <div className="bg-white border-b border-gray-100">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex space-x-8 overflow-x-auto">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium text-sm whitespace-nowrap transition-colors ${
                      activeTab === tab.id
                        ? 'border-green-500 text-green-600'
                        : 'border-transparent text-gray-500 hover:text-green-600 hover:border-green-300'
                    }`}
                  >
                    {tab.icon}
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div> */}

          {/* Navigation */}
          <nav className="flex-1 p-4">
            <div className="space-y-1">
              {tabs.map((item) => (
                <Button
                  key={item.id}
                  variant={activeTab === item.id ? "default" : "ghost"}
                  className={`w-full justify-start transition-all duration-200 ${
                    activeTab === item.id 
                      ? 'bg-green-600 text-white hover:bg-green-700 shadow-sm' 
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
            {activeTab === 'home' && (
              <div className="space-y-6">
                {/* Welcome Header with Risk Badge */}
                <div className="rounded-xl p-8 bg-gray-50">
                  <div className="flex items-center justify-between">
                    <div>
                      <h1 className="text-3xl font-bold text-gray-900 mb-2">
                        Good morning, {mockPatient.firstName}!
                      </h1>
                      <p className="text-gray-600 text-lg">
                        Ready to make today a healthy day?
                      </p>
                    </div>
                    <div className="text-right text-sm">
                      <Badge className="bg-green-50 text-green-700 border-green-200">
                        Health Score: {mockPatient.riskScore}/100
                      </Badge>
                      <div className="text-green-600">
                        {mockPatient.riskLevel} Risk Level
                      </div>
                    </div>
                  </div>
                </div>

                {/* Quick Health Overview */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <Card className="rounded-lg">
                    <CardContent className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                          <Heart className="w-5 h-5 text-red-600" />
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">Blood Pressure</p>
                          <p className="font-semibold">{mockPatient.vitals.bloodPressure}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="rounded-lg">
                    <CardContent className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center">
                          <Droplets className="w-5 h-5 text-purple-600" />
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">Blood Sugar</p>
                          <p className="font-semibold">{mockPatient.vitals.bloodSugar} mg/dL</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="rounded-lg">
                    <CardContent className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                          <Activity className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">Steps Today</p>
                          <p className="font-semibold">{mockPatient.vitals.steps.toLocaleString()}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="rounded-lg">
                    <CardContent className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                          <Target className="w-5 h-5 text-green-600" />
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">Daily Progress</p>
                          <p className="font-semibold">{Math.round(progressPercentage)}% Complete</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Today's Priority Tasks */}
                <Card className="rounded-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="w-5 h-5" />
                      Today's Priority Tasks
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {mockDailyPlan.filter((task: any) => task.importance === 'critical' || task.importance === 'high').slice(0, 3).map((task: any) => (
                        <div key={task.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                          <Checkbox
                            checked={completedTasks.has(task.id)}
                            onCheckedChange={() => toggleTask(task.id)}
                          />
                          <div className="flex items-center gap-2">
                            {task.icon}
                            <span className={`font-medium ${completedTasks.has(task.id) ? 'line-through text-gray-500' : 'text-gray-900'}`}>
                              {task.title}
                            </span>
                          </div>
                          <Badge 
                            variant={task.importance === 'critical' ? 'destructive' : 'default'}
                            className="text-xs ml-auto"
                          >
                            {task.importance.toUpperCase()}
                          </Badge>
                        </div>
                      ))}
                    </div>
                    <Button className="w-full mt-4" onClick={() => setActiveTab('daily-plan')}>
                      View Full Daily Plan
                    </Button>
                  </CardContent>
                </Card>

                {/* Streaks & Achievements */}
                {/* <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card className="rounded-sm">
                    <CardContent className="p-4 text-center">
                      <div className="w-12 h-12 bg-orange-500 rounded-sm flex items-center justify-center mx-auto mb-2">
                        <Pill className="w-6 h-6 text-white" />
                      </div>
                      <p className="text-sm text-gray-600">Medication Streak</p>
                      <p className="text-2xl font-bold text-orange-700">{mockPatient.streaks.medication} days</p>
                    </CardContent>
                  </Card>

                  <Card className="rounded-sm bg-gradient-to-br from-green-50 to-green-100">
                    <CardContent className="p-4 text-center">
                      <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-2">
                        <Dumbbell className="w-6 h-6 text-white" />
                      </div>
                      <p className="text-sm text-gray-600">Exercise Streak</p>
                      <p className="text-2xl font-bold text-green-700">{mockPatient.streaks.exercise} days</p>
                    </CardContent>
                  </Card>

                  <Card className="rounded-lg bg-gradient-to-br from-blue-50 to-blue-100">
                    <CardContent className="p-4 text-center">
                      <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-2">
                        <Stethoscope className="w-6 h-6 text-white" />
                      </div>
                      <p className="text-sm text-gray-600">Vitals Logging</p>
                      <p className="text-2xl font-bold text-blue-700">{mockPatient.streaks.vitalsLogging} days</p>
                    </CardContent>
                  </Card>
                </div> */}
              </div>
            )}

            {activeTab === 'onboarding' && (
              <div className="space-y-6">
                <div className="text-center">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome to Your Health Journey</h2>
                  <p className="text-gray-600">Let's get you set up for success with personalized care</p>
                </div>

                {/* Onboarding Progress */}
                <Card className="rounded-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <CheckCircle className="w-5 h-5" />
                      Setup Progress
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center gap-4 p-4 bg-green-50 rounded-lg border border-green-200">
                        <CheckCircle className="w-6 h-6 text-green-600" />
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900">Profile Setup</h3>
                          <p className="text-sm text-gray-600">Basic information and medical history completed</p>
                        </div>
                      </div>

                      <div className="flex items-center gap-4 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                        <Clock className="w-6 h-6 text-yellow-600" />
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900">Connect Health Devices</h3>
                          <p className="text-sm text-gray-600">Link your blood pressure monitor and glucose meter</p>
                        </div>
                        <Button size="sm">Connect</Button>
                      </div>

                      <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <Plus className="w-6 h-6 text-gray-400" />
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900">EHR Integration (Optional)</h3>
                          <p className="text-sm text-gray-600">Connect with your doctor's electronic health records</p>
                        </div>
                        <Button variant="outline" size="sm">Setup Later</Button>
                      </div>

                      <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <Bell className="w-6 h-6 text-gray-400" />
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900">Notification Preferences</h3>
                          <p className="text-sm text-gray-600">Set up medication and appointment reminders</p>
                        </div>
                        <Button variant="outline" size="sm">Configure</Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Consent & Privacy */}
                <Card className="rounded-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="w-5 h-5" />
                      Privacy & Consent
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="p-4 bg-blue-50 rounded-lg">
                        <h3 className="font-semibold text-gray-900 mb-2">Your Data is Secure</h3>
                        <p className="text-sm text-gray-600 mb-3">
                          We use bank-level encryption and comply with HIPAA regulations to protect your health information.
                        </p>
                        <div className="space-y-2">
                          <label className="flex items-center gap-2">
                            <Checkbox defaultChecked />
                            <span className="text-sm">I consent to sharing my data with my healthcare providers</span>
                          </label>
                          <label className="flex items-center gap-2">
                            <Checkbox />
                            <span className="text-sm">I agree to receive wellness tips and educational content</span>
                          </label>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'health' && (
              <div className="space-y-6">
                <div className="text-center space-y-2">
                  <h2 className="text-2xl font-bold text-gray-900">Your Health Status</h2>
                  <p className="text-gray-600">Your heart health is stable today. Keep up the good work!</p>
                </div>

                {/* Risk Status */}
                <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-lg">
                  <CardContent className="p-8 text-center">
                    <div className="inline-flex items-center space-x-3 bg-amber-50 border border-amber-200 rounded-lg px-6 py-4">
                      <Shield className="w-8 h-8 text-amber-600" />
                      <div className="text-left">
                        <div className="font-semibold text-amber-800">Medium Risk</div>
                        <div className="text-sm text-amber-600">Score: {mockPatient.riskScore}/100</div>
                      </div>
                    </div>
                    <p className="mt-4 text-gray-600 max-w-md mx-auto">
                      Your current health metrics show you're managing your conditions well. Small daily improvements will help lower your risk over time.
                    </p>
                  </CardContent>
                </Card>

                {/* Today's Vitals */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <Card className="bg-white border border-gray-200">
                    <CardContent className="p-6 text-center">
                      <Heart className="w-8 h-8 text-green-600 mx-auto mb-3" />
                      <div className="text-2xl font-bold text-gray-900">{mockPatient.vitals.bloodPressure}</div>
                      <div className="text-sm text-gray-600">Blood Pressure</div>
                      <div className="text-xs text-green-600 mt-1">Monitor closely</div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white border border-gray-200">
                    <CardContent className="p-6 text-center">
                      <Droplets className="w-8 h-8 text-green-600 mx-auto mb-3" />
                      <div className="text-2xl font-bold text-gray-900">{mockPatient.vitals.bloodSugar} mg/dL</div>
                      <div className="text-sm text-gray-600">Blood Glucose</div>
                      <div className="text-xs text-green-600 mt-1">Within range</div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white border border-gray-200">
                    <CardContent className="p-6 text-center">
                      <Activity className="w-8 h-8 text-green-600 mx-auto mb-3" />
                      <div className="text-2xl font-bold text-gray-900">{mockPatient.vitals.steps.toLocaleString()}</div>
                      <div className="text-sm text-gray-600">Steps Today</div>
                      <div className="text-xs text-green-600 mt-1">Great progress!</div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white border border-gray-200">
                    <CardContent className="p-6 text-center">
                      <Moon className="w-8 h-8 text-green-600 mx-auto mb-3" />
                      <div className="text-2xl font-bold text-gray-900">{mockPatient.vitals.sleep}h</div>
                      <div className="text-sm text-gray-600">Sleep Last Night</div>
                      <div className="text-xs text-green-600 mt-1">Well rested</div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}

            {activeTab === 'daily-plan' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">Your Daily Wellness Plan</h2>
                    <p className="text-gray-600">Personalized tasks to help you reach your health goals</p>
                  </div>
                  <Badge variant="outline" className="text-sm">
                    {completedCount} of {mockDailyPlan.length} completed
                  </Badge>
                </div>

                {/* Progress Overview */}
                <Card className="bg-white border border-gray-200">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="font-semibold text-gray-900">Today's Progress</h3>
                        <p className="text-sm text-gray-600">Keep up the great work!</p>
                      </div>
                      <div className="text-right">
                        <div className="text-3xl font-bold text-green-600">{Math.round(progressPercentage)}%</div>
                        <div className="text-sm text-gray-500">Complete</div>
                      </div>
                    </div>
                    <Progress value={progressPercentage} className="h-4" />
                  </CardContent>
                </Card>

                {/* Task Cards with Streaks */}
                <div className="space-y-4">
                  {mockDailyPlan.map((task: any) => (
                    <Card key={task.id} className={`bg-white border border-gray-200 transition-all duration-200 ${
                      completedTasks.has(task.id) ? 'ring-2 ring-green-200 bg-green-50/30' : 'hover:shadow-md'
                    }`}>
                      <CardContent className="p-6">
                        <div className="flex items-start gap-4">
                          <Checkbox
                            checked={completedTasks.has(task.id)}
                            onCheckedChange={() => toggleTask(task.id)}
                            className="mt-1"
                          />
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              {task.icon}
                              <h3 className={`font-semibold ${completedTasks.has(task.id) ? 'text-green-800 line-through' : 'text-gray-900'}`}>
                                {task.title}
                              </h3>
                              <Badge 
                                variant={task.importance === 'critical' ? 'destructive' : 'outline'}
                                className="text-xs"
                              >
                                {task.importance}
                              </Badge>
                              {task.streak > 0 && (
                                <Badge variant="outline" className="text-xs bg-green-50 text-green-700">
                                  🔥 {task.streak} day streak
                                </Badge>
                              )}
                            </div>
                            <p className="text-sm text-gray-600 mb-3">{task.description}</p>
                            <div className="flex items-center gap-4 text-xs text-gray-500">
                              <div className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {task.time}
                              </div>
                              <div className="flex items-center gap-1">
                                <Target className="w-3 h-3" />
                                {task.category}
                              </div>
                            </div>
                          </div>
                          {completedTasks.has(task.id) && (
                            <CheckCircle className="w-6 h-6 text-green-500" />
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'vitals' && (
              <div className="space-y-6">
                <div className="text-center">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Log Your Vitals</h2>
                  <p className="text-gray-600">Track your health metrics to monitor progress</p>
                </div>

                {/* Quick Entry Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="bg-white border border-gray-200">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Heart className="w-5 h-5 text-green-600" />
                        Blood Pressure
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="text-sm text-gray-600">Systolic</label>
                            <Input placeholder="120" className="mt-1" />
                          </div>
                          <div>
                            <label className="text-sm text-gray-600">Diastolic</label>
                            <Input placeholder="80" className="mt-1" />
                          </div>
                        </div>
                        <p className="text-xs text-gray-500">Last reading: {mockPatient.vitals.lastBP}</p>
                        <Button className="w-full">
                          <Plus className="w-4 h-4 mr-2" />
                          Log Reading
                        </Button>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white border border-gray-200">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Droplets className="w-5 h-5 text-green-600" />
                        Blood Glucose
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div>
                          <label className="text-sm text-gray-600">Glucose Level (mg/dL)</label>
                          <Input placeholder="100" className="mt-1" />
                        </div>
                        <div>
                          <label className="text-sm text-gray-600">Timing</label>
                          <select className="w-full mt-1 p-2 border rounded-md text-sm">
                            <option>Before meal</option>
                            <option>After meal</option>
                            <option>Bedtime</option>
                            <option>Random</option>
                          </select>
                        </div>
                        <p className="text-xs text-gray-500">Last reading: {mockPatient.vitals.lastGlucose}</p>
                        <Button className="w-full">
                          <Plus className="w-4 h-4 mr-2" />
                          Log Reading
                        </Button>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white border border-gray-200">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <User className="w-5 h-5 text-green-600" />
                        Weight
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div>
                          <label className="text-sm text-gray-600">Weight (lbs)</label>
                          <Input placeholder="165" className="mt-1" />
                        </div>
                        <div className="text-xs text-gray-500">
                          Goal: {mockPatient.goals.weightTarget} lbs
                        </div>
                        <Button className="w-full">
                          <Plus className="w-4 h-4 mr-2" />
                          Log Weight
                        </Button>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white border border-gray-200">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Activity className="w-5 h-5 text-green-600" />
                        Activity & Steps
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div>
                          <label className="text-sm text-gray-600">Steps Today</label>
                          <Input placeholder="8420" className="mt-1" />
                        </div>
                        <div>
                          <label className="text-sm text-gray-600">Exercise Minutes</label>
                          <Input placeholder="30" className="mt-1" />
                        </div>
                        <div className="text-xs text-gray-500">
                          Goal: {mockPatient.goals.dailySteps.toLocaleString()} steps/day
                        </div>
                        <Button className="w-full">
                          <Plus className="w-4 h-4 mr-2" />
                          Log Activity
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Recent Readings */}
                <Card className="bg-white border border-gray-200">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-green-600" />
                      Recent Readings
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <Heart className="w-4 h-4 text-green-600" />
                          <span className="font-medium">Blood Pressure: 142/89</span>
                        </div>
                        <span className="text-sm text-gray-500">Today, 6:30 PM</span>
                      </div>
                      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <Droplets className="w-4 h-4 text-green-600" />
                          <span className="font-medium">Glucose: 156 mg/dL</span>
                        </div>
                        <span className="text-sm text-gray-500">Today, 2:20 PM</span>
                      </div>
                      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <User className="w-4 h-4 text-green-600" />
                          <span className="font-medium">Weight: 165 lbs</span>
                        </div>
                        <span className="text-sm text-gray-500">Yesterday, 7:00 AM</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'medications' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">Your Medications</h2>
                    <p className="text-gray-600">Manage your prescriptions and reminders</p>
                  </div>
                  <Button>
                    <Plus className="w-4 h-4 mr-2" />
                    Add Medication
                  </Button>
                </div>

                {/* Today's Medications */}
                <Card className="rounded-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Clock className="w-5 h-5 text-blue-500" />
                      Today's Schedule
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {mockMedications.map((med: any) => (
                        <div key={med.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                          <div className="flex items-center gap-4">
                            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                              <Pill className="w-6 h-6 text-blue-600" />
                            </div>
                            <div>
                              <h3 className="font-semibold text-gray-900">{med.name}</h3>
                              <p className="text-sm text-gray-600">{med.dosage} • {med.frequency}</p>
                              <p className="text-xs text-gray-500">Next dose: {med.nextDose}</p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant={med.taken ? "default" : "outline"}>
                              {med.taken ? "Taken" : "Pending"}
                            </Badge>
                            {!med.taken && (
                              <Button size="sm">Mark Taken</Button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Medication Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {mockMedications.map((med: any) => (
                    <Card key={`detail-${med.id}`} className="bg-white border border-gray-200">
                      <CardHeader>
                        <CardTitle className="flex items-center justify-between">
                          <span>{med.name}</span>
                          <Badge variant="outline">{med.condition}</Badge>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          <div>
                            <h4 className="font-medium text-gray-900 mb-1">Dosage & Schedule</h4>
                            <p className="text-sm text-gray-600">{med.dosage}, {med.frequency}</p>
                          </div>
                          
                          {med.sideEffects && (
                            <div>
                              <h4 className="font-medium text-gray-900 mb-1">Common Side Effects</h4>
                              <p className="text-sm text-gray-600">{med.sideEffects}</p>
                            </div>
                          )}
                          
                          {med.interactions && (
                            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                              <h4 className="font-medium text-gray-800 mb-1 flex items-center gap-1">
                                <AlertTriangle className="w-4 h-4" />
                                Drug Interactions
                              </h4>
                              <p className="text-sm text-gray-700">{med.interactions}</p>
                            </div>
                          )}
                          
                          <div className="flex justify-between items-center pt-2">
                            <span className="text-sm text-gray-500">Prescribed by Dr. {med.prescriber}</span>
                            <Button variant="outline" size="sm">
                              <Settings className="w-4 h-4 mr-1" />
                              Edit
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                {/* Adherence Tracking */}
                <Card className="bg-white border border-gray-200">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-green-600" />
                      Medication Adherence
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">This Week</span>
                        <span className="font-semibold text-green-600">92% Adherence</span>
                      </div>
                      <Progress value={92} className="h-3" />
                      <div className="grid grid-cols-7 gap-2 mt-4">
                        {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, index) => (
                          <div key={day} className="text-center">
                            <div className="text-xs text-gray-500 mb-1">{day}</div>
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium ${
                              index < 6 ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-400'
                            }`}>
                              {index < 6 ? '✓' : '?'}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'progress' && (
              <div className="space-y-6">
                <div className="text-center">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Your Health Progress</h2>
                  <p className="text-gray-600">Track your journey towards better health</p>
                </div>

                {/* Progress Overview Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <Card className="bg-white border border-gray-200">
                    <CardContent className="p-6 text-center">
                      <Target className="w-8 h-8 text-green-600 mx-auto mb-3" />
                      <div className="text-2xl font-bold text-gray-900">85%</div>
                      <div className="text-sm text-gray-600">Goals Achieved</div>
                      <div className="text-xs text-gray-500 mt-1">This month</div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-white border border-gray-200">
                    <CardContent className="p-6 text-center">
                      <TrendingUp className="w-8 h-8 text-green-600 mx-auto mb-3" />
                      <div className="text-2xl font-bold text-gray-900">-8 lbs</div>
                      <div className="text-sm text-gray-600">Weight Change</div>
                      <div className="text-xs text-gray-500 mt-1">Since starting</div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-white border border-gray-200">
                    <CardContent className="p-6 text-center">
                      <Award className="w-8 h-8 text-green-600 mx-auto mb-3" />
                      <div className="text-2xl font-bold text-gray-900">12</div>
                      <div className="text-sm text-gray-600">Day Streak</div>
                      <div className="text-xs text-gray-500 mt-1">Daily tasks</div>
                    </CardContent>
                  </Card>
                </div>

                {/* Health Metrics Chart */}
                <Card className="bg-white border border-gray-200">
                  <CardHeader>
                    <CardTitle>Health Metrics Trends</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
                      <div className="text-center text-gray-500">
                        <BarChart3 className="w-12 h-12 mx-auto mb-2 text-green-600" />
                        <p>Interactive charts would be displayed here</p>
                        <p className="text-sm">Blood pressure, glucose, weight trends</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Achievements */}
                <Card className="bg-white border border-gray-200">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Award className="w-5 h-5 text-green-600" />
                      Recent Achievements
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                        <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                          <Award className="w-5 h-5 text-green-600" />
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900">7-Day Streak!</h4>
                          <p className="text-sm text-gray-600">Completed all daily tasks for a week</p>
                        </div>
                        <span className="text-xs text-gray-500 ml-auto">2 days ago</span>
                      </div>
                      
                      <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                        <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                          <Target className="w-5 h-5 text-green-600" />
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900">Blood Pressure Goal</h4>
                          <p className="text-sm text-gray-600">Maintained healthy BP for 5 days</p>
                        </div>
                        <span className="text-xs text-gray-500 ml-auto">1 week ago</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'coach' && (
              <div className="space-y-6">
                <div className="text-center">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Your AI Health Coach</h2>
                  <p className="text-gray-600">Get personalized guidance and support 24/7</p>
                </div>

                {/* Chat Interface */}
                <Card className="bg-white border border-gray-200">
                  <CardHeader className="border-b">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-green-600 rounded-full flex items-center justify-center">
                        <MessageCircle className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <CardTitle>Dr. AI Assistant</CardTitle>
                        <p className="text-sm text-gray-600">Your personal health coach</p>
                      </div>
                      <div className="ml-auto">
                        <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-0">
                    {/* Chat Messages */}
                    <div className="h-96 overflow-y-auto p-6 space-y-4">
                      <div className="flex gap-3">
                        <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
                          <MessageCircle className="w-4 h-4 text-white" />
                        </div>
                        <div className="bg-gray-100 rounded-lg p-3 max-w-xs">
                          <p className="text-sm">Good morning, Sarah! I noticed your blood pressure reading from yesterday was 142/89. How are you feeling today?</p>
                          <span className="text-xs text-gray-500 mt-1 block">9:15 AM</span>
                        </div>
                      </div>

                      <div className="flex gap-3 justify-end">
                        <div className="bg-green-600 text-white rounded-lg p-3 max-w-xs">
                          <p className="text-sm">I'm feeling okay, just a bit tired. Should I be concerned about the reading?</p>
                          <span className="text-xs text-green-200 mt-1 block">9:18 AM</span>
                        </div>
                        <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center flex-shrink-0">
                          <User className="w-4 h-4 text-gray-600" />
                        </div>
                      </div>

                      <div className="flex gap-3">
                        <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
                          <MessageCircle className="w-4 h-4 text-white" />
                        </div>
                        <div className="bg-gray-100 rounded-lg p-3 max-w-md">
                          <p className="text-sm">That reading is slightly elevated. Let's focus on your medication adherence and stress management. Have you taken your morning Lisinopril? Also, try some deep breathing exercises - I can guide you through them.</p>
                          <span className="text-xs text-gray-500 mt-1 block">9:20 AM</span>
                        </div>
                      </div>

                      <div className="flex gap-3 justify-end">
                        <div className="bg-green-600 text-white rounded-lg p-3 max-w-xs">
                          <p className="text-sm">Yes, I took it with breakfast. The breathing exercises sound good!</p>
                          <span className="text-xs text-green-200 mt-1 block">9:22 AM</span>
                        </div>
                        <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center flex-shrink-0">
                          <User className="w-4 h-4 text-gray-600" />
                        </div>
                      </div>
                    </div>

                    {/* Chat Input */}
                    <div className="border-t p-4">
                      <div className="flex gap-2">
                        <Input 
                          placeholder="Ask me anything about your health..." 
                          className="flex-1"
                        />
                        <Button>
                          <Send className="w-4 h-4" />
                        </Button>
                      </div>
                      
                      {/* Quick Actions */}
                      <div className="flex gap-2 mt-3">
                        <Button variant="outline" size="sm">
                          <Heart className="w-4 h-4 mr-1" />
                          BP Help
                        </Button>
                        <Button variant="outline" size="sm">
                          <Pill className="w-4 h-4 mr-1" />
                          Medication
                        </Button>
                        <Button variant="outline" size="sm">
                          <Activity className="w-4 h-4 mr-1" />
                          Exercise
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* AI Insights */}
                <Card className="bg-white border border-gray-200">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Brain className="w-5 h-5 text-green-600" />
                      AI Health Insights
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <h4 className="font-medium text-gray-900 mb-2">Personalized Recommendation</h4>
                        <p className="text-sm text-gray-700">Based on your recent vitals, consider increasing your daily walk to 45 minutes. This could help improve your blood pressure control.</p>
                      </div>
                      
                      <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                        <h4 className="font-medium text-green-900 mb-2">Positive Trend</h4>
                        <p className="text-sm text-green-800">Great job! Your medication adherence has improved by 15% this month. Keep up the excellent work!</p>
                      </div>
                      
                      <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <h4 className="font-medium text-gray-900 mb-2">Gentle Reminder</h4>
                        <p className="text-sm text-gray-700">Don't forget to log your evening blood pressure reading. Consistent monitoring helps us track your progress better.</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'timeline' && (
              <div className="space-y-6">
                <div className="text-center space-y-2">
                  <h2 className="text-2xl font-bold text-gray-900">Your Progress This Week</h2>
                  <p className="text-gray-600">See how your health metrics are trending</p>
                </div>

                <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-lg">
                  <CardContent className="p-8">
                    <div className="grid grid-cols-7 gap-4">
                      {mockTimelineData.map((day, index) => (
                        <div key={index} className="text-center space-y-3">
                          <div className="text-sm font-medium text-gray-600">{day.date}</div>
                          
                          {/* Glucose */}
                          <div className="space-y-1">
                            <div className={`w-12 h-12 mx-auto rounded-lg flex items-center justify-center text-xs font-medium ${
                              day.glucose < 150 ? 'bg-emerald-100 text-emerald-800' : 
                              day.glucose < 160 ? 'bg-amber-100 text-amber-800' : 
                              'bg-red-100 text-red-800'
                            }`}>
                              {day.glucose}
                            </div>
                            <div className="text-xs text-gray-500">Glucose</div>
                          </div>

                          {/* Steps */}
                          <div className="space-y-1">
                            <div className={`w-8 h-8 mx-auto rounded-full flex items-center justify-center text-xs ${
                              day.steps > 8000 ? 'bg-blue-500' : 'bg-gray-300'
                            }`}>
                              {day.steps > 8000 ? '✓' : '○'}
                            </div>
                            <div className="text-xs text-gray-500">Steps</div>
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    <div className="mt-6 p-4 bg-emerald-50 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <TrendingUp className="w-5 h-5 text-emerald-600" />
                        <span className="font-medium text-emerald-800">Great progress!</span>
                      </div>
                      <p className="text-sm text-emerald-700 mt-1">
                        Your glucose levels have improved by 3% this week. Keep up the healthy habits!
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'reminders' && (
              <div className="space-y-6">
                <div className="text-center space-y-2">
                  <h2 className="text-2xl font-bold text-gray-900">Reminders & Notifications</h2>
                  <p className="text-gray-600">Stay on track with your health routine</p>
                </div>

                <div className="space-y-4">
                  {mockReminders.map((reminder) => (
                    <Card key={reminder.id} className={`bg-white/80 backdrop-blur-sm border-0 shadow-sm ${
                      reminder.urgent ? 'ring-2 ring-amber-200' : ''
                    }`}>
                      <CardContent className="p-6">
                        <div className="flex items-center space-x-4">
                          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                            reminder.type === 'medication' ? 'bg-blue-100' :
                            reminder.type === 'appointment' ? 'bg-purple-100' :
                            'bg-emerald-100'
                          }`}>
                            {reminder.type === 'medication' ? <Pill className="w-5 h-5 text-blue-600" /> :
                             reminder.type === 'appointment' ? <Calendar className="w-5 h-5 text-purple-600" /> :
                             <Zap className="w-5 h-5 text-emerald-600" />}
                          </div>
                          <div className="flex-1">
                            <h3 className="font-semibold text-gray-900">{reminder.title}</h3>
                            <p className="text-sm text-gray-600">{reminder.message}</p>
                          </div>
                          <div className="text-right">
                            <div className="text-sm font-medium text-gray-900">{reminder.time}</div>
                            {reminder.urgent && (
                              <Badge className="bg-amber-100 text-amber-800 text-xs mt-1">Urgent</Badge>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'insights' && (
              <div className="space-y-6">
                <div className="text-center space-y-2">
                  <h2 className="text-2xl font-bold text-gray-900">AI Health Insights</h2>
                  <p className="text-gray-600">Understanding what affects your health today</p>
                </div>

                <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-lg">
                  <CardContent className="p-8">
                    <div className="flex items-center space-x-3 mb-6">
                      <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                        <Brain className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900">Today's Health Factors</h3>
                        <p className="text-sm text-gray-600">AI analysis of your wellness patterns</p>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="p-4 bg-emerald-50 rounded-lg border border-emerald-200">
                        <div className="flex items-center space-x-2 mb-2">
                          <CheckCircle className="w-5 h-5 text-emerald-600" />
                          <span className="font-medium text-emerald-800">Positive Impact</span>
                        </div>
                        <p className="text-sm text-emerald-700">
                          Your 7+ hours of sleep and morning walk are helping keep your blood pressure stable today.
                        </p>
                      </div>

                      <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
                        <div className="flex items-center space-x-2 mb-2">
                          <AlertCircle className="w-5 h-5 text-amber-600" />
                          <span className="font-medium text-amber-800">Watch This</span>
                        </div>
                        <p className="text-sm text-amber-700">
                          Your glucose level is slightly elevated. Consider having a lighter lunch and taking that afternoon walk.
                        </p>
                      </div>

                      <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                        <div className="flex items-center space-x-2 mb-2">
                          <TrendingUp className="w-5 h-5 text-blue-600" />
                          <span className="font-medium text-blue-800">Trending Well</span>
                        </div>
                        <p className="text-sm text-blue-700">
                          Your medication adherence this week is excellent! This consistency is helping improve your overall health.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'chat' && (
              <div className="space-y-6">
                <div className="text-center space-y-2">
                  <h2 className="text-2xl font-bold text-gray-900">Chat with Your Care Team</h2>
                  <p className="text-gray-600">Secure messaging with Dr. Chen and your healthcare team</p>
                </div>

                <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-lg">
                  <CardContent className="p-8">
                    <div className="space-y-4 mb-6 max-h-64 overflow-y-auto">
                      {/* Sample messages */}
                      <div className="flex items-start space-x-3">
                        <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                          <User className="w-4 h-4 text-white" />
                        </div>
                        <div className="bg-blue-50 rounded-lg p-3 max-w-xs">
                          <p className="text-sm text-blue-900">
                            Hi Sarah! I reviewed your latest readings. Your progress is looking great. Keep up the good work with your morning walks!
                          </p>
                          <p className="text-xs text-blue-600 mt-1">Dr. Chen • 2 hours ago</p>
                        </div>
                      </div>

                      <div className="flex items-start space-x-3 justify-end">
                        <div className="bg-emerald-50 rounded-lg p-3 max-w-xs">
                          <p className="text-sm text-emerald-900">
                            Thank you! I've been feeling much better. Should I continue with the same medication schedule?
                          </p>
                          <p className="text-xs text-emerald-600 mt-1">You • 1 hour ago</p>
                        </div>
                        <div className="w-8 h-8 bg-emerald-500 rounded-full flex items-center justify-center">
                          <User className="w-4 h-4 text-white" />
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      <Input
                        placeholder="Type your message to the care team..."
                        value={chatMessage}
                        onChange={(e) => setChatMessage(e.target.value)}
                        className="flex-1"
                      />
                      <Button className="bg-emerald-600 hover:bg-emerald-700">
                        <Send className="w-4 h-4" />
                      </Button>
                    </div>

                    <p className="text-xs text-gray-500 mt-2">
                      💬 Messages are secure and HIPAA compliant. Response time: Usually within 4 hours during business hours.
                    </p>
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
