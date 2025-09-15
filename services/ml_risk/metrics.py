"""Model evaluation metrics for cardiometabolic risk prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, accuracy_score, f1_score, precision_score, recall_score,
    calibration_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import structlog

logger = structlog.get_logger()


class ModelMetrics:
    """Comprehensive model evaluation metrics for healthcare ML."""
    
    def __init__(self):
        pass
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        try:
            metrics = {}
            
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
            
            # ROC and Precision-Recall metrics
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            
            # Calibration metrics
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            
            # Calculate calibration slope
            cal_fraction_pos, cal_mean_pred = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            if len(cal_fraction_pos) > 1:
                # Linear regression slope between observed and predicted
                metrics['calibration_slope'] = np.corrcoef(
                    cal_mean_pred, cal_fraction_pos
                )[0, 1] if len(cal_mean_pred) > 1 else 1.0
            else:
                metrics['calibration_slope'] = 1.0
            
            # Sensitivity and Specificity
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Likelihood ratios
            metrics['positive_likelihood_ratio'] = (
                metrics['sensitivity'] / (1 - metrics['specificity'])
                if metrics['specificity'] < 1 else float('inf')
            )
            metrics['negative_likelihood_ratio'] = (
                (1 - metrics['sensitivity']) / metrics['specificity']
                if metrics['specificity'] > 0 else float('inf')
            )
            
            # Net Benefit Analysis (Clinical Decision Making)
            thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
            for threshold in thresholds:
                net_benefit = self.calculate_net_benefit(y_true, y_pred_proba, threshold)
                metrics[f'net_benefit_{int(threshold*100)}'] = net_benefit
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to calculate metrics", error=str(e))
            return {}
    
    def calculate_net_benefit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                            threshold: float) -> float:
        """Calculate net benefit for clinical decision analysis."""
        try:
            # Net benefit = (TP/n) - (FP/n) * (threshold/(1-threshold))
            n = len(y_true)
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
            
            tp = np.sum((y_true == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true == 0) & (y_pred_binary == 1))
            
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            return net_benefit
            
        except Exception as e:
            logger.error("Failed to calculate net benefit", error=str(e))
            return 0.0
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Generate detailed classification report."""
        try:
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Add confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            report['confusion_matrix'] = {
                'true_negative': int(cm[0, 0]),
                'false_positive': int(cm[0, 1]),
                'false_negative': int(cm[1, 0]),
                'true_positive': int(cm[1, 1])
            }
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate classification report", error=str(e))
            return {}
    
    def calculate_fairness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_pred_proba: np.ndarray, 
                                 sensitive_attributes: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate fairness metrics across demographic groups."""
        try:
            fairness_metrics = {}
            
            for attr_name, attr_values in sensitive_attributes.items():
                fairness_metrics[attr_name] = {}
                unique_values = np.unique(attr_values)
                
                for value in unique_values:
                    mask = attr_values == value
                    if np.sum(mask) > 10:  # Minimum group size
                        group_metrics = self.calculate_all_metrics(
                            y_true[mask], y_pred[mask], y_pred_proba[mask]
                        )
                        fairness_metrics[attr_name][f'group_{value}'] = group_metrics
                
                # Calculate fairness gaps
                if len(fairness_metrics[attr_name]) >= 2:
                    groups = list(fairness_metrics[attr_name].keys())
                    for metric in ['roc_auc', 'average_precision', 'accuracy']:
                        values = [
                            fairness_metrics[attr_name][group].get(metric, 0)
                            for group in groups
                        ]
                        fairness_metrics[attr_name][f'{metric}_gap'] = max(values) - min(values)
            
            return fairness_metrics
            
        except Exception as e:
            logger.error("Failed to calculate fairness metrics", error=str(e))
            return {}
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      save_path: str = None) -> plt.Figure:
        """Plot ROC curve."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc = roc_auc_score(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return plt.gcf()
            
        except Exception as e:
            logger.error("Failed to plot ROC curve", error=str(e))
            return None
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_path: str = None) -> plt.Figure:
        """Plot Precision-Recall curve."""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AP = {avg_precision:.3f})')
            
            # Baseline (prevalence)
            baseline = np.mean(y_true)
            plt.axhline(y=baseline, color='navy', linestyle='--',
                       label=f'Baseline (prevalence = {baseline:.3f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return plt.gcf()
            
        except Exception as e:
            logger.error("Failed to plot PR curve", error=str(e))
            return None
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                             save_path: str = None) -> plt.Figure:
        """Plot calibration curve."""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            
            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                    color='darkorange', label='Model', linewidth=2, markersize=8)
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
            
            plt.ylabel("Fraction of positives")
            plt.xlabel("Mean predicted probability")
            plt.title("Calibration Plot")
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return plt.gcf()
            
        except Exception as e:
            logger.error("Failed to plot calibration curve", error=str(e))
            return None
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: str = None) -> plt.Figure:
        """Plot confusion matrix."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Low Risk', 'High Risk'],
                       yticklabels=['Low Risk', 'High Risk'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return plt.gcf()
            
        except Exception as e:
            logger.error("Failed to plot confusion matrix", error=str(e))
            return None
    
    def generate_model_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_pred_proba: np.ndarray, 
                            feature_names: List[str] = None,
                            sensitive_attributes: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """Generate comprehensive model evaluation report."""
        try:
            report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'sample_size': len(y_true),
                'prevalence': float(np.mean(y_true)),
                'performance_metrics': self.calculate_all_metrics(y_true, y_pred, y_pred_proba),
                'classification_report': self.generate_classification_report(y_true, y_pred)
            }
            
            # Add fairness metrics if sensitive attributes provided
            if sensitive_attributes:
                report['fairness_metrics'] = self.calculate_fairness_metrics(
                    y_true, y_pred, y_pred_proba, sensitive_attributes
                )
            
            # Clinical interpretation thresholds
            report['clinical_thresholds'] = {
                'low_risk': 0.15,
                'moderate_risk': 0.30,
                'high_risk': 0.50
            }
            
            # Risk distribution
            risk_bins = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']
            risk_counts = pd.cut(
                y_pred_proba, 
                bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], 
                labels=risk_bins
            ).value_counts().to_dict()
            
            report['risk_distribution'] = {str(k): int(v) for k, v in risk_counts.items()}
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate model report", error=str(e))
            return {}


def calculate_clinical_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                             cost_matrix: Dict[str, float] = None) -> Dict[str, float]:
    """Calculate clinical decision-making metrics."""
    if cost_matrix is None:
        cost_matrix = {
            'true_positive_benefit': 1.0,    # Benefit of correctly identifying high risk
            'false_positive_cost': 0.2,      # Cost of false alarm
            'false_negative_cost': 5.0,      # Cost of missing high risk patient
            'true_negative_benefit': 0.1     # Small benefit of correctly identifying low risk
        }
    
    metrics = {}
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    best_threshold = 0.5
    best_utility = -float('inf')
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate expected utility
        utility = (
            tp * cost_matrix['true_positive_benefit'] +
            tn * cost_matrix['true_negative_benefit'] -
            fp * cost_matrix['false_positive_cost'] -
            fn * cost_matrix['false_negative_cost']
        )
        
        if utility > best_utility:
            best_utility = utility
            best_threshold = threshold
        
        metrics[f'utility_threshold_{threshold:.2f}'] = utility
    
    metrics['optimal_threshold'] = best_threshold
    metrics['optimal_utility'] = best_utility
    
    return metrics
