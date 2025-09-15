import React from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ChevronDown, Bell, Settings, User, Heart, Search } from 'lucide-react';

interface Patient {
  id: string;
  name: string;
  age: number;
  mrn: string;
  riskLevel: 'Low' | 'Medium' | 'High';
}

interface TopBarProps {
  currentPatient: Patient;
  clinicianName: string;
  onPatientSelect?: () => void;
  onNotifications?: () => void;
  onSettings?: () => void;
  className?: string;
}

const TopBar: React.FC<TopBarProps> = ({
  currentPatient,
  clinicianName,
  onPatientSelect,
  onNotifications,
  onSettings,
  className = ""
}) => {
  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'Low':
        return 'bg-emerald-100 text-emerald-800 border-emerald-200';
      case 'Medium':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      case 'High':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className={`bg-white border-b border-gray-200 px-6 py-4 ${className}`}>
      <div className="flex items-center justify-between">
        {/* Left side - Logo and Patient Selector */}
        <div className="flex items-center space-x-6">
          {/* Logo */}
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-emerald-600 rounded-lg flex items-center justify-center">
              <Heart className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">CareSight</span>
          </div>

          {/* Patient Selector */}
          <div className="flex items-center space-x-3">
            <div className="h-6 w-px bg-gray-300"></div>
            <Button
              variant="ghost"
              onClick={onPatientSelect}
              className="flex items-center space-x-3 px-4 py-2 h-auto hover:bg-gray-50 rounded-xl"
            >
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-emerald-500 rounded-full flex items-center justify-center text-white font-semibold text-sm">
                {currentPatient.name.split(' ').map(n => n[0]).join('')}
              </div>
              <div className="text-left">
                <div className="font-semibold text-gray-900 text-sm">
                  {currentPatient.name}
                </div>
                <div className="text-xs text-gray-500">
                  Age {currentPatient.age} • MRN: {currentPatient.mrn}
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Badge className={`${getRiskLevelColor(currentPatient.riskLevel)} border text-xs font-medium`}>
                  {currentPatient.riskLevel} Risk
                </Badge>
                <ChevronDown className="w-4 h-4 text-gray-400" />
              </div>
            </Button>
          </div>
        </div>

        {/* Right side - Search, Notifications, and Profile */}
        <div className="flex items-center space-x-4">
          {/* Search */}
          <Button variant="ghost" size="sm" className="text-gray-600 hover:text-gray-900">
            <Search className="w-5 h-5" />
          </Button>

          {/* Notifications */}
          <Button
            variant="ghost"
            size="sm"
            onClick={onNotifications}
            className="relative text-gray-600 hover:text-gray-900"
          >
            <Bell className="w-5 h-5" />
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></div>
          </Button>

          {/* Settings */}
          <Button
            variant="ghost"
            size="sm"
            onClick={onSettings}
            className="text-gray-600 hover:text-gray-900"
          >
            <Settings className="w-5 h-5" />
          </Button>

          {/* Clinician Profile */}
          <div className="flex items-center space-x-3 pl-4 border-l border-gray-200">
            <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
              <User className="w-5 h-5 text-gray-600" />
            </div>
            <div className="text-sm">
              <div className="font-medium text-gray-900">{clinicianName}</div>
              <div className="text-xs text-gray-500">Clinician</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TopBar;
