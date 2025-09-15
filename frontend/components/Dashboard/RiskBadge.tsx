import React from 'react';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface RiskBadgeProps {
  condition: string;
  score: number;
  level: 'Low' | 'Medium' | 'High';
  trend?: 'up' | 'down' | 'stable';
  trendValue?: string;
  icon: React.ReactNode;
  className?: string;
}

const RiskBadge: React.FC<RiskBadgeProps> = ({
  condition,
  score,
  level,
  trend,
  trendValue,
  icon,
  className = ""
}) => {
  const getLevelColors = (level: string) => {
    switch (level) {
      case 'Low':
        return {
          bg: 'bg-emerald-50',
          border: 'border-emerald-200',
          text: 'text-emerald-800',
          badge: 'bg-emerald-100 text-emerald-800',
          score: 'text-emerald-600'
        };
      case 'Medium':
        return {
          bg: 'bg-amber-50',
          border: 'border-amber-200',
          text: 'text-amber-800',
          badge: 'bg-amber-100 text-amber-800',
          score: 'text-amber-600'
        };
      case 'High':
        return {
          bg: 'bg-red-50',
          border: 'border-red-200',
          text: 'text-red-800',
          badge: 'bg-red-100 text-red-800',
          score: 'text-red-600'
        };
      default:
        return {
          bg: 'bg-gray-50',
          border: 'border-gray-200',
          text: 'text-gray-800',
          badge: 'bg-gray-100 text-gray-800',
          score: 'text-gray-600'
        };
    }
  };

  const colors = getLevelColors(level);

  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-red-500" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-emerald-500" />;
      case 'stable':
        return <Minus className="w-4 h-4 text-gray-500" />;
      default:
        return null;
    }
  };

  return (
    <div className={`${colors.bg} ${colors.border} border-2 rounded-2xl p-6 transition-all duration-200 hover:shadow-md ${className}`}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`w-10 h-10 ${colors.score} bg-white rounded-xl flex items-center justify-center shadow-sm`}>
            {icon}
          </div>
          <div>
            <h3 className={`font-semibold ${colors.text} text-lg`}>{condition}</h3>
            <Badge className={`${colors.badge} text-xs font-medium`}>
              {level} Risk
            </Badge>
          </div>
        </div>
        <div className="text-right">
          <div className={`text-3xl font-bold ${colors.score} mb-1`}>{score}</div>
          <div className="text-xs text-gray-500 uppercase tracking-wide">Risk Score</div>
        </div>
      </div>
      
      {trend && trendValue && (
        <div className="flex items-center space-x-2 pt-3 border-t border-white/50">
          {getTrendIcon()}
          <span className="text-sm text-gray-600 font-medium">{trendValue}</span>
          <span className="text-xs text-gray-500">vs last month</span>
        </div>
      )}
    </div>
  );
};

export default RiskBadge;
