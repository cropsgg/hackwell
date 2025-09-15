import React from 'react';
import { TrendingUp, TrendingDown, AlertCircle, CheckCircle } from 'lucide-react';

interface TimelineEvent {
  date: string;
  type: 'assessment' | 'intervention' | 'milestone' | 'alert';
  title: string;
  description: string;
  value?: number;
  trend?: 'up' | 'down' | 'stable';
}

interface TimelineChartProps {
  events: TimelineEvent[];
  className?: string;
}

const TimelineChart: React.FC<TimelineChartProps> = ({
  events,
  className = ""
}) => {
  const getEventIcon = (type: string, trend?: string) => {
    switch (type) {
      case 'assessment':
        return <div className="w-3 h-3 bg-blue-500 rounded-full" />;
      case 'intervention':
        return <CheckCircle className="w-4 h-4 text-emerald-600" />;
      case 'milestone':
        return <div className="w-3 h-3 bg-emerald-500 rounded-full" />;
      case 'alert':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <div className="w-3 h-3 bg-gray-400 rounded-full" />;
    }
  };

  const getEventColors = (type: string) => {
    switch (type) {
      case 'assessment':
        return {
          bg: 'bg-blue-50',
          border: 'border-blue-200',
          text: 'text-blue-800'
        };
      case 'intervention':
        return {
          bg: 'bg-emerald-50',
          border: 'border-emerald-200',
          text: 'text-emerald-800'
        };
      case 'milestone':
        return {
          bg: 'bg-emerald-50',
          border: 'border-emerald-200',
          text: 'text-emerald-800'
        };
      case 'alert':
        return {
          bg: 'bg-red-50',
          border: 'border-red-200',
          text: 'text-red-800'
        };
      default:
        return {
          bg: 'bg-gray-50',
          border: 'border-gray-200',
          text: 'text-gray-800'
        };
    }
  };

  const getTrendIcon = (trend?: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-red-500" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-emerald-500" />;
      default:
        return null;
    }
  };

  return (
    <div className={`bg-white rounded-xl border-2 border-gray-200 p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Patient Timeline</h3>
        <div className="text-sm text-gray-500">Last 6 months</div>
      </div>

      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-200"></div>

        <div className="space-y-6">
          {events.map((event, index) => {
            const colors = getEventColors(event.type);
            return (
              <div key={index} className="relative flex items-start space-x-4">
                {/* Timeline dot */}
                <div className="relative z-10 flex items-center justify-center w-12 h-12 bg-white border-2 border-gray-200 rounded-full">
                  {getEventIcon(event.type, event.trend)}
                </div>

                {/* Event content */}
                <div className="flex-1 min-w-0">
                  <div className={`${colors.bg} ${colors.border} border rounded-xl p-4 transition-all duration-200 hover:shadow-sm`}>
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <h4 className={`font-semibold ${colors.text} text-base`}>
                          {event.title}
                        </h4>
                        <p className="text-gray-700 text-sm mt-1 leading-relaxed">
                          {event.description}
                        </p>
                      </div>
                      <div className="flex items-center space-x-2 ml-4 flex-shrink-0">
                        {event.trend && getTrendIcon(event.trend)}
                        {event.value && (
                          <span className="text-lg font-bold text-gray-900">
                            {event.value}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="text-xs text-gray-500 font-medium">
                      {event.date}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default TimelineChart;
