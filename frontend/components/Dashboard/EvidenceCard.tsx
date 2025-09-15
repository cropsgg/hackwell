import React from 'react';
import { Badge } from '@/components/ui/badge';
import { ExternalLink, Shield, BookOpen } from 'lucide-react';

interface EvidenceCardProps {
  title: string;
  description: string;
  evidenceLevel: 'A' | 'B' | 'C';
  source: string;
  url?: string;
  type: 'guideline' | 'study' | 'research';
  className?: string;
}

const EvidenceCard: React.FC<EvidenceCardProps> = ({
  title,
  description,
  evidenceLevel,
  source,
  url,
  type,
  className = ""
}) => {
  const getTypeIcon = () => {
    switch (type) {
      case 'guideline':
        return <Shield className="w-5 h-5 text-emerald-600" />;
      case 'study':
        return <BookOpen className="w-5 h-5 text-blue-600" />;
      case 'research':
        return <BookOpen className="w-5 h-5 text-purple-600" />;
      default:
        return <BookOpen className="w-5 h-5 text-gray-600" />;
    }
  };

  const getEvidenceLevelColor = (level: string) => {
    switch (level) {
      case 'A':
        return 'bg-emerald-100 text-emerald-800 border-emerald-200';
      case 'B':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'C':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getCardBorder = () => {
    switch (type) {
      case 'guideline':
        return 'border-emerald-200 hover:border-emerald-300';
      case 'study':
        return 'border-blue-200 hover:border-blue-300';
      case 'research':
        return 'border-purple-200 hover:border-purple-300';
      default:
        return 'border-gray-200 hover:border-gray-300';
    }
  };

  return (
    <div className={`bg-white border-2 ${getCardBorder()} rounded-xl p-5 transition-all duration-200 hover:shadow-lg group ${className}`}>
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0 mt-1">
          {getTypeIcon()}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between mb-3">
            <h3 className="font-semibold text-gray-900 text-lg leading-tight pr-2">
              {title}
            </h3>
            <Badge className={`${getEvidenceLevelColor(evidenceLevel)} border text-xs font-bold flex-shrink-0`}>
              Level {evidenceLevel}
            </Badge>
          </div>
          
          <p className="text-gray-700 text-sm leading-relaxed mb-4">
            {description}
          </p>
          
          <div className="flex items-center justify-between">
            <div className="text-xs text-gray-500 font-medium">
              {source}
            </div>
            {url && (
              <button className="flex items-center space-x-1 text-xs text-blue-600 hover:text-blue-700 transition-colors group-hover:underline">
                <span>View Source</span>
                <ExternalLink className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EvidenceCard;
