"""
How to Use This Structure
1. For running on a server/locally:
   * Use run_evaluation.py with command-line arguments
2. Example: python run_evaluation.py --data_path=/path/to/data --batch_size=32
"""
# Import standard libraries
import os
import logging
import argparse
import sys
from tqdm import tqdm

# Import from our utility modules
from utils.config import Config
from utils.data_utils import load_and_prepare_data
from models.xray_models import load_models
from evaluation.metrics import evaluate_predictions
from evaluation.visualization import plot_results, plot_roc_curves, plot_pr_curves
from utils.inference import run_inference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("xray_evaluation")

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()
    
    # Update config if arguments provided
    if args.data_path:
        Config.data_path = args.data_path
        Config.image_path = os.path.join(args.data_path, "images")
        Config.csv_path = os.path.join(args.data_path, "Data_Entry_2017_v2020.csv")
    
    if args.output_path:
        Config.output_path = args.output_path
    
    if args.batch_size:
        Config.batch_size = args.batch_size
        
    if args.samples_per_disease:
        Config.samples_per_disease = args.samples_per_disease
        
    if args.seed:
        Config.random_seed = args.seed
    
    # Create output directory
    os.makedirs(Config.output_path, exist_ok=True)
    
    try:
        # Load models
        model_mimic, model_chex, pathologies = load_models(Config.device)
        
        # Prepare data
        test_df, label_indices = load_and_prepare_data(Config, pathologies)
        
        # Run inference
        y_true, y_pred = run_inference(test_df, model_mimic, model_chex, label_indices, Config)
        
        # Evaluate and report results
        results_df = evaluate_predictions(
            y_true, y_pred, Config.target_labels, Config.output_path
        )
        
        # Create visualizations
        plot_results(results_df, test_df, Config.target_labels, Config.output_path)
        plot_roc_curves(y_true, y_pred, Config.target_labels, Config.output_path)
        plot_pr_curves(y_true, y_pred, Config.target_labels, Config.output_path)
        
        logger.info("Evaluation completed successfully")
        logger.info(f"Results saved to {Config.output_path}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":