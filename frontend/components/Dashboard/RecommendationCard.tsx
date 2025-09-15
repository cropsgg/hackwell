import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { CheckCircle, Clock, Apple, Activity, Pill, Calendar } from 'lucide-react';

interface RecommendationCardProps {
  title: string;
  description: string;
  category: 'diet' | 'exercise' | 'medication' | 'lifestyle';
  priority: 'high' | 'medium' | 'low';
  timeframe: string;
  completed?: boolean;
  onToggle?: () => void;
  className?: string;
}

const RecommendationCard: React.FC<RecommendationCardProps> = ({
  title,
  description,
  category,
  priority,
  timeframe,
  completed = false,
  onToggle,
  className = ""
}) => {
  const getCategoryConfig = () => {
    switch (category) {
      case 'diet':
        return {
          icon: <Apple className="w-5 h-5" />,
          color: 'text-emerald-600',
          bg: 'bg-emerald-50',
          border: 'border-emerald-200'
        };
      case 'exercise':
        return {
          icon: <Activity className="w-5 h-5" />,
          color: 'text-blue-600',
          bg: 'bg-blue-50',
          border: 'border-blue-200'
        };
      case 'medication':
        return {
          icon: <Pill className="w-5 h-5" />,
          color: 'text-purple-600',
          bg: 'bg-purple-50',
          border: 'border-purple-200'
        };
      case 'lifestyle':
        return {
          icon: <Calendar className="w-5 h-5" />,
          color: 'text-amber-600',
          bg: 'bg-amber-50',
          border: 'border-amber-200'
        };
      default:
        return {
          icon: <Clock className="w-5 h-5" />,
          color: 'text-gray-600',
          bg: 'bg-gray-50',
          border: 'border-gray-200'
        };
    }
  };

  const getPriorityColor = () => {
    switch (priority) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'medium':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      case 'low':
        return 'bg-emerald-100 text-emerald-800 border-emerald-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const config = getCategoryConfig();

  return (
    <div className={`bg-white border-2 ${config.border} rounded-xl p-5 transition-all duration-200 hover:shadow-md ${completed ? 'opacity-75' : ''} ${className}`}>
      <div className="flex items-start space-x-4">
        <div className={`w-10 h-10 ${config.bg} rounded-xl flex items-center justify-center ${config.color} flex-shrink-0 mt-1`}>
          {config.icon}
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between mb-3">
            <h3 className={`font-semibold text-gray-900 text-lg leading-tight ${completed ? 'line-through' : ''}`}>
              {title}
            </h3>
            <div className="flex items-center space-x-2 flex-shrink-0 ml-3">
              <Badge className={`${getPriorityColor()} border text-xs font-medium`}>
                {priority.toUpperCase()}
              </Badge>
            </div>
          </div>
          
          <p className="text-gray-700 text-sm leading-relaxed mb-4">
            {description}
          </p>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-xs text-gray-500">
              <Clock className="w-4 h-4" />
              <span className="font-medium">{timeframe}</span>
            </div>
            
            {onToggle && (
              <Button
                variant={completed ? "outline" : "default"}
                size="sm"
                onClick={onToggle}
                className={`text-xs ${completed ? 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100' : 'bg-emerald-600 hover:bg-emerald-700 text-white'}`}
              >
                {completed ? (
                  <>
                    <CheckCircle className="w-4 h-4 mr-1" />
                    Completed
                  </>
                ) : (
                  'Mark Done'
                )}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendationCard;
