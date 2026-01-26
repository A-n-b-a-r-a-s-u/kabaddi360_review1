"""
Evaluation Script for Kabaddi Injury Prediction System
Computes AUC, Sensitivity, Specificity, and Event-level Accuracy
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
from loguru import logger


class InjuryPredictionEvaluator:
    """Evaluate injury prediction system performance."""
    
    def __init__(self, ground_truth_path: str, predictions_path: str):
        """
        Initialize evaluator.
        
        Args:
            ground_truth_path: Path to ground truth labels CSV
            predictions_path: Path to predictions JSON from pipeline
        """
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.predictions = self._load_predictions(predictions_path)
        
        logger.info(f"Loaded {len(self.ground_truth)} ground truth labels")
        logger.info(f"Loaded {len(self.predictions)} predictions")
    
    def _load_ground_truth(self, path: str) -> pd.DataFrame:
        """
        Load ground truth labels.
        
        Expected CSV format:
        frame_num, injury_occurred, injury_severity, event_type
        """
        try:
            df = pd.read_csv(path)
            required_cols = ["frame_num", "injury_occurred"]
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Ground truth CSV must contain: {required_cols}")
            
            return df
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            raise
    
    def _load_predictions(self, path: str) -> Dict:
        """Load predictions from pipeline summary JSON."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Extract risk timeline
            if "risk" in data and "timeline" in data["risk"]:
                return data["risk"]["timeline"]
            else:
                raise ValueError("Predictions JSON missing 'risk.timeline'")
        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")
            raise
    
    def align_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align ground truth and predictions by frame number.
        
        Returns:
            (y_true, y_scores, y_pred) arrays
        """
        y_true = []
        y_scores = []
        
        # Create prediction lookup
        pred_dict = {p["frame"]: p["risk_score"] for p in self.predictions}
        
        for _, row in self.ground_truth.iterrows():
            frame = row["frame_num"]
            
            if frame in pred_dict:
                y_true.append(int(row["injury_occurred"]))
                y_scores.append(pred_dict[frame])
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Convert scores to binary predictions (threshold at 70 for high risk)
        y_pred = (y_scores >= 70).astype(int)
        
        logger.info(f"Aligned {len(y_true)} samples")
        return y_true, y_scores, y_pred
    
    def calculate_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate Area Under ROC Curve."""
        try:
            auc = roc_auc_score(y_true, y_scores)
            logger.info(f"AUC: {auc:.4f}")
            return auc
        except Exception as e:
            logger.error(f"Failed to calculate AUC: {e}")
            return 0.0
    
    def calculate_sensitivity_specificity(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate sensitivity and specificity."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        logger.info(f"Sensitivity (Recall): {sensitivity:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        
        return sensitivity, specificity
    
    def calculate_event_level_accuracy(self) -> Dict:
        """
        Calculate event-level accuracy for falls and impacts.
        
        Checks if predicted events align with ground truth events.
        """
        # Extract event frames from ground truth
        gt_events = set(
            self.ground_truth[
                self.ground_truth["injury_occurred"] == 1
            ]["frame_num"].values
        )
        
        # Extract high-risk frames from predictions
        pred_events = set(
            p["frame"] for p in self.predictions 
            if p["risk_score"] >= 70
        )
        
        # Calculate metrics
        true_positives = len(gt_events & pred_events)
        false_positives = len(pred_events - gt_events)
        false_negatives = len(gt_events - pred_events)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        logger.info(f"Event-level Precision: {precision:.4f}")
        logger.info(f"Event-level Recall: {recall:.4f}")
        logger.info(f"Event-level F1: {f1:.4f}")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, save_path: str):
        """Plot and save ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Injury Prediction')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray, save_path: str):
        """Plot and save Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Injury Prediction')
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {save_path}")
        plt.close()
    
    def generate_report(self, output_dir: str) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_dir: Directory to save report and plots
        
        Returns:
            Dictionary with all metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info("EVALUATION REPORT")
        logger.info("="*60)
        
        # Align data
        y_true, y_scores, y_pred = self.align_data()
        
        # Calculate metrics
        auc = self.calculate_auc(y_true, y_scores)
        sensitivity, specificity = self.calculate_sensitivity_specificity(y_true, y_pred)
        event_metrics = self.calculate_event_level_accuracy()
        
        # Classification report
        logger.info("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["No Injury", "Injury"]))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Plot curves
        self.plot_roc_curve(y_true, y_scores, str(output_path / "roc_curve.png"))
        self.plot_precision_recall_curve(y_true, y_scores, str(output_path / "precision_recall_curve.png"))
        
        # Compile report
        report = {
            "auc": float(auc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "event_level": event_metrics,
            "confusion_matrix": cm.tolist(),
            "total_samples": len(y_true),
            "positive_samples": int(np.sum(y_true)),
            "negative_samples": int(len(y_true) - np.sum(y_true))
        }
        
        # Save report
        report_path = output_path / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nEvaluation report saved to {report_path}")
        logger.info("="*60)
        
        return report


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Kabaddi Injury Prediction System")
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth CSV file"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to pipeline_summary.json from pipeline output"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = InjuryPredictionEvaluator(args.ground_truth, args.predictions)
    report = evaluator.generate_report(args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"AUC: {report['auc']:.4f}")
    print(f"Sensitivity: {report['sensitivity']:.4f}")
    print(f"Specificity: {report['specificity']:.4f}")
    print(f"Event-level F1: {report['event_level']['f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
