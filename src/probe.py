import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--safe_path', type=str, required=True)
    parser.add_argument('--harmful_path', type=str, required=True)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    
    # Load representations
    X_safe = np.load(args.safe_path)
    X_harmful = np.load(args.harmful_path)
    
    # Create labels (0=safe, 1=harmful)
    y_safe = np.zeros(len(X_safe))
    y_harmful = np.ones(len(X_harmful))
    
    # Combine data
    X = np.vstack([X_safe, X_harmful])
    y = np.concatenate([y_safe, y_harmful])
    
    # Train classifier with cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=42)
    
    # Get multiple metrics via cross_validate
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(clf, X, y, cv=args.cv, scoring=scoring)
    
    # Calculate means and stds
    acc_mean = cv_results['test_accuracy'].mean()
    acc_std = cv_results['test_accuracy'].std()
    prec_mean = cv_results['test_precision'].mean()
    rec_mean = cv_results['test_recall'].mean()
    f1_mean = cv_results['test_f1'].mean()
    
    # Print results
    print(f"Layer {args.layer if args.layer else 'N/A'}:")
    print(f"  Accuracy:  {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Precision: {prec_mean:.4f}")
    print(f"  Recall:    {rec_mean:.4f}")
    print(f"  F1:        {f1_mean:.4f}")
    
    # Save to CSV if output file specified
    if args.output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        # Check if file exists AND has content (not just header)
        file_exists = os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0
        
        with open(args.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header if file doesn't exist or is empty
            if not file_exists:
                writer.writerow(['layer', 'accuracy', 'std', 'precision', 'recall', 'f1'])
            
            writer.writerow([
                args.layer if args.layer is not None else 'N/A',
                f"{acc_mean:.4f}",
                f"{acc_std:.4f}",
                f"{prec_mean:.4f}",
                f"{rec_mean:.4f}",
                f"{f1_mean:.4f}"
            ])
        
        print(f"\nResults appended to: {args.output_file}")

if __name__ == "__main__":
    main()