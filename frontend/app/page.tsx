"use client";

import React from 'react';
import { 
  Eye, 
  Heart, 
  Brain, 
  Activity, 
  Users, 
  Zap, 
  Target, 
  Globe,
  ArrowRight,
  Play,
  Shield,
  Clock,
  TrendingUp,
  Award,
  CheckCircle,
  Stethoscope,
  Pill,
  Apple,
  Star
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useRouter } from 'next/navigation';

export default function Homepage() {
  const router = useRouter();
  const healthMetrics = [
    {
      score: 92,
      title: "Diabetes Risk",
      subtitle: "Low Risk",
      trend: "+5% improvement",
      icon: <Pill className="w-6 h-6" />,
      color: "bg-emerald-500",
      image: "https://images.pexels.com/photos/3768131/pexels-photo-3768131.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&fit=crop"
    },
    {
      score: 88,
      title: "Heart Health",
      subtitle: "Excellent", 
      trend: "Stable",
      icon: <Heart className="w-6 h-6" />,
      color: "bg-red-500",
      image: "https://images.pexels.com/photos/3768997/pexels-photo-3768997.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&fit=crop"
    },
    {
      score: 85,
      title: "Blood Pressure",
      subtitle: "Optimal",
      trend: "-3% this month",
      icon: <Activity className="w-6 h-6" />,
      color: "bg-blue-500",
      image: "https://images.pexels.com/photos/3771069/pexels-photo-3771069.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&fit=crop"
    }
  ];

  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: "AI-Powered Risk Assessment",
      description: "Advanced machine learning algorithms analyze your health data to predict risks for diabetes, hypertension, and heart disease before symptoms appear.",
      color: "bg-purple-100 text-purple-700"
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Evidence-Based Recommendations", 
      description: "Get personalized treatment plans backed by the latest medical research and clinical guidelines from leading healthcare institutions.",
      color: "bg-emerald-100 text-emerald-700"
    },
    {
      icon: <TrendingUp className="w-8 h-8" />,
      title: "Continuous Health Monitoring",
      description: "Track your health metrics over time with intelligent insights that help you understand patterns and make informed decisions.",
      color: "bg-blue-100 text-blue-700"
    },
    {
      icon: <Stethoscope className="w-8 h-8" />,
      title: "Clinical-Grade Analysis",
      description: "Hospital-quality health assessments powered by the same AI technology used by healthcare professionals worldwide.",
      color: "bg-cyan-100 text-cyan-700"
    }
  ];

  const testimonials = [
    {
      name: "Dr. Sarah Chen",
      role: "Cardiologist, Mayo Clinic",
      content: "CareSight has revolutionized how we approach preventive care. The AI insights help us identify at-risk patients months before traditional methods.",
      avatar: "https://images.pexels.com/photos/5215024/pexels-photo-5215024.jpeg?auto=compress&cs=tinysrgb&w=150&h=150&fit=crop"
    },
    {
      name: "Michael Rodriguez",
      role: "Type 2 Diabetes Patient",
      content: "Thanks to CareSight's early detection, I was able to prevent diabetes through lifestyle changes. The personalized recommendations were life-changing.",
      avatar: "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=150&h=150&fit=crop"
    },
    {
      name: "Dr. James Wilson",
      role: "Endocrinologist",
      content: "The predictive accuracy for chronic disease risk is remarkable. It's like having a crystal ball for patient health outcomes.",
      avatar: "https://images.pexels.com/photos/5452201/pexels-photo-5452201.jpeg?auto=compress&cs=tinysrgb&w=150&h=150&fit=crop"
    }
  ];

  const trustIndicators = [
    { icon: <Award className="w-6 h-6" />, text: "FDA Breakthrough Device" },
    { icon: <Shield className="w-6 h-6" />, text: "HIPAA Compliant" },
    { icon: <CheckCircle className="w-6 h-6" />, text: "Clinical Validation" },
    { icon: <Users className="w-6 h-6" />, text: "500K+ Patients" }
  ];

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="absolute top-0 left-0 right-0 z-50 px-6 py-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-white rounded-md flex items-center justify-center">
              <Heart className="w-5 h-5 text-emerald-600" />
            </div>
            <span className="text-white font-semibold text-lg">CareSight</span>
          </div>
          
          <div className="hidden md:flex items-center space-x-8 text-white/90">
            <a href="#platform" className="hover:text-white transition-colors">Platform</a>
            <a href="#research" className="hover:text-white transition-colors">Research</a>
            <a href="#about" className="hover:text-white transition-colors">About</a>
            <a href="#contact" className="hover:text-white transition-colors">Contact</a>
          </div>

          <Button onClick={() => {router.push('/dashboard')}} className="bg-white text-gray-900 hover:bg-gray-100">
            Get Started
          </Button>
        </div>
      </nav>

            {/* Hero Section */}
      <div className="relative h-screen flex items-center">
        {/* Background Image */}
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{
            backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), url('https://images.pexels.com/photos/3768131/pexels-photo-3768131.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&fit=crop')`
          }}
        />
        
        <div className="relative z-10 max-w-7xl mx-auto px-6">
          <div className="max-w-2xl">
            <h1 className="text-6xl md:text-7xl font-bold text-white mb-6 leading-tight">
              See Beyond.
              <br />
              Unlock Your Health
            </h1>
            
            <p className="text-xl text-white/90 mb-8 leading-relaxed">
              Your body holds the answers — we help you see them.
            </p>

            <div className="flex items-center space-x-4">
              <Button size="lg" className="bg-white text-gray-900 hover:bg-gray-100 px-8 py-4 text-lg">
                Start Your Journey
                <ArrowRight className="w-5 h-5 ml-2" />
              </Button>
              <Button size="lg" variant="outline" className="border-white text-white hover:bg-white hover:text-gray-900 px-8 py-4 text-lg">
                <Play className="w-5 h-5 mr-2" />
                Watch Demo
              </Button>
            </div>
          </div>
        </div>

        {/* Floating Health Cards - Enhanced */}
        <div className="absolute bottom-8 left-6 space-y-6 hidden lg:block z-20">
          <div className="flex items-center space-x-3 text-white">
            <div className="w-10 h-10 bg-emerald-500/20 rounded-full flex items-center justify-center">
              <Activity className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <div className="text-sm font-medium">Real-Time Analysis</div>
              <div className="text-xs text-white/70">Instant health insights powered by AI</div>
            </div>
          </div>

          <div className="flex items-center space-x-3 text-white">
            <div className="w-10 h-10 bg-blue-500/20 rounded-full flex items-center justify-center">
              <Target className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <div className="text-sm font-medium">Personalized Care Plans</div>
              <div className="text-xs text-white/70">Tailored recommendations for your unique health profile</div>
            </div>
          </div>

          <div className="flex items-center space-x-3 text-white">
            <div className="w-10 h-10 bg-red-500/20 rounded-full flex items-center justify-center">
              <Heart className="w-5 h-5 text-red-400" />
            </div>
            <div>
              <div className="text-sm font-medium">Chronic Disease Prevention</div>
              <div className="text-xs text-white/70">Early detection for diabetes, hypertension & heart disease</div>
            </div>
          </div>
        </div>
      </div>

      {/* Trust Indicators */}
      <section className="py-8 bg-gray-50 border-b">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-wrap items-center justify-center gap-8 text-sm text-gray-600">
            {trustIndicators.map((indicator, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div className="text-emerald-600">
                  {indicator.icon}
                </div>
                <span className="font-medium">{indicator.text}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Platform Description - Enhanced */}
      <section id="platform" className="py-24 bg-gray-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="max-w-4xl">
            <Badge className="mb-6 bg-emerald-100 text-emerald-800 hover:bg-emerald-100">
              AI-Powered Healthcare
            </Badge>
            <h2 className="text-5xl md:text-6xl font-bold text-gray-900 mb-8 leading-tight">
              Predictive Health Intelligence for 
              <span className="text-emerald-600"> Chronic Disease Prevention</span>
            </h2>
            <p className="text-xl text-gray-600 leading-relaxed">
              Our AI analyzes over 290 biomarkers and lifestyle factors to predict your risk of diabetes, 
              hypertension, and heart disease—often years before symptoms appear. Get personalized, 
              evidence-based recommendations that help you take control of your health journey.
            </p>
          </div>
        </div>
      </section>

      {/* Enhanced Health Metrics Cards */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">
              Real Patient Health Insights
            </h3>
            <p className="text-lg text-gray-600">
              See how our AI transforms complex health data into actionable insights
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {healthMetrics.map((metric, index) => (
              <Card key={index} className="overflow-hidden group hover:shadow-xl transition-all duration-300 border-0 shadow-lg">
                <div className="relative h-64">
                  <img 
                    src={metric.image}
                    alt={metric.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />
                  
                  {/* Enhanced Score Badge */}
                  <div className="absolute top-4 left-4">
                    <div className="bg-white rounded-xl px-4 py-3 shadow-lg">
                      <div className="flex items-center space-x-2">
                        <div className={`w-8 h-8 ${metric.color} rounded-lg flex items-center justify-center text-white`}>
                          {metric.icon}
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-gray-900">{metric.score}</div>
                          <div className="text-xs text-gray-500 uppercase tracking-wide">Score</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Enhanced Content */}
                  <div className="absolute bottom-4 left-4 right-4 text-white">
                    <h3 className="text-xl font-semibold mb-1">{metric.title}</h3>
                    <p className="text-white/90 mb-2">{metric.subtitle}</p>
                    <div className="flex items-center space-x-2">
                      <TrendingUp className="w-4 h-4 text-emerald-400" />
                      <span className="text-sm text-emerald-400">{metric.trend}</span>
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Enhanced Features Section */}
      <section className="py-24 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <Badge className="mb-6 bg-blue-100 text-blue-800 hover:bg-blue-100">
              Advanced AI Technology
            </Badge>
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              Healthcare Innovation at Your Fingertips
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Powered by cutting-edge machine learning and validated by clinical research, 
              our platform delivers hospital-grade health assessments in real-time.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            {features.map((feature, index) => (
              <div key={index} className="group">
                <div className="flex items-start space-x-4">
                  <div className={`w-16 h-16 ${feature.color} rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300`}>
                    {feature.icon}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-gray-900 mb-3">{feature.title}</h3>
                    <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-24 bg-gray-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              Trusted by Healthcare Professionals
            </h2>
            <p className="text-xl text-gray-600">
              See what doctors and patients are saying about CareSight 
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <Card key={index} className="p-6 border-0 shadow-lg hover:shadow-xl transition-shadow duration-300">
                <CardContent className="p-0">
                  <div className="flex items-center mb-4">
                    {[...Array(5)].map((_, i) => (
                      <Star key={i} className="w-4 h-4 text-yellow-400 fill-current" />
                    ))}
                  </div>
                  <p className="text-gray-600 mb-6 italic">"{testimonial.content}"</p>
                  <div className="flex items-center space-x-3">
                    <img 
                      src={testimonial.avatar} 
                      alt={testimonial.name}
                      className="w-12 h-12 rounded-full object-cover"
                    />
                    <div>
                      <div className="font-semibold text-gray-900">{testimonial.name}</div>
                      <div className="text-sm text-gray-500">{testimonial.role}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-blue-900">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Ready to Transform Your Health?
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            Join thousands of people and practitioners who are already using our platform to unlock better health outcomes.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4">
            <Button size="lg" className="bg-white text-blue-900 hover:bg-gray-100 px-8 py-4 text-lg">
              Start Free Trial
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
            <Button size="lg" variant="outline" className="border-white text-white hover:bg-white hover:text-blue-900 px-8 py-4 text-lg">
              Schedule Demo
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="md:col-span-1">
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-blue-600 rounded-md flex items-center justify-center">
                  <Heart className="w-5 h-5 text-white" />
                </div>
                <span className="font-semibold text-lg">CareSight</span>
              </div>
              <p className="text-gray-400 text-sm leading-relaxed">
                Empowering better health outcomes through advanced analytics and personalized insights.
              </p>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Platform</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Dashboard</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Analytics</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Reports</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Integrations</a></li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Resources</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
                <li><a href="#" className="hover:text-white transition-colors">API Reference</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Research</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Support</a></li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">About</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Privacy</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Terms</a></li>
              </ul>
            </div>
          </div>

          <div className="border-t border-gray-800 mt-12 pt-8 flex flex-col md:flex-row items-center justify-between">
            <p className="text-gray-400 text-sm">
              © 2025 HealthFlow. All rights reserved.
            </p>
            <div className="flex items-center space-x-6 mt-4 md:mt-0">
              <a href="#" className="text-gray-400 hover:text-white transition-colors">
                <Users className="w-5 h-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition-colors">
                <Activity className="w-5 h-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition-colors">
                <Globe className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
