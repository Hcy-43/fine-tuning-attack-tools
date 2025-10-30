import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def plot_layer_sweep(input_file, output_file='layer_sweep_plot.png'):
    # Read results
    df = pd.read_csv(input_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Layer Sweep Results: Llama2-7b Harmful Content Classification', 
                 fontsize=16, fontweight='bold')
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: Accuracy with error bars
    ax1 = axes[0, 0]
    ax1.errorbar(df['layer'], df['accuracy'], yerr=df['std'], 
                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_xlabel('Layer Number', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Layer', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
    best_acc_idx = df['accuracy'].idxmax()
    ax1.axvline(x=df.loc[best_acc_idx, 'layer'], color='g', 
                linestyle='--', alpha=0.5, label=f"Best: Layer {df.loc[best_acc_idx, 'layer']}")
    ax1.legend()
    
    # Plot 2: Precision and Recall
    ax2 = axes[0, 1]
    ax2.plot(df['layer'], df['precision'], marker='s', linewidth=2, 
             markersize=8, label='Precision')
    ax2.plot(df['layer'], df['recall'], marker='^', linewidth=2, 
             markersize=8, label='Recall')
    ax2.set_xlabel('Layer Number', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Precision & Recall vs Layer', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: F1 Score
    ax3 = axes[1, 0]
    ax3.plot(df['layer'], df['f1'], marker='D', linewidth=2, 
             markersize=8, color='purple')
    ax3.set_xlabel('Layer Number', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.set_title('F1 Score vs Layer', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    best_f1_idx = df['f1'].idxmax()
    ax3.axvline(x=df.loc[best_f1_idx, 'layer'], color='g', 
                linestyle='--', alpha=0.5, label=f"Best: Layer {df.loc[best_f1_idx, 'layer']}")
    ax3.legend()
    
    # Plot 4: All metrics together (normalized)
    ax4 = axes[1, 1]
    ax4.plot(df['layer'], df['accuracy'], marker='o', linewidth=2, 
             markersize=6, label='Accuracy', alpha=0.7)
    ax4.plot(df['layer'], df['precision'], marker='s', linewidth=2, 
             markersize=6, label='Precision', alpha=0.7)
    ax4.plot(df['layer'], df['recall'], marker='^', linewidth=2, 
             markersize=6, label='Recall', alpha=0.7)
    ax4.plot(df['layer'], df['f1'], marker='D', linewidth=2, 
             markersize=6, label='F1', alpha=0.7)
    ax4.set_xlabel('Layer Number', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nBest Accuracy: {df['accuracy'].max():.4f} at Layer {df.loc[df['accuracy'].idxmax(), 'layer']}")
    print(f"Best F1 Score: {df['f1'].max():.4f} at Layer {df.loc[df['f1'].idxmax(), 'layer']}")
    print(f"Best Precision: {df['precision'].max():.4f} at Layer {df.loc[df['precision'].idxmax(), 'layer']}")
    print(f"Best Recall: {df['recall'].max():.4f} at Layer {df.loc[df['recall'].idxmax(), 'layer']}")
    print(f"\nWorst Accuracy: {df['accuracy'].min():.4f} at Layer {df.loc[df['accuracy'].idxmin(), 'layer']}")
    print(f"\nAccuracy Range: {df['accuracy'].max() - df['accuracy'].min():.4f}")
    print("\n" + "="*60)
    
    # Show top 3 layers by F1
    print("\nTop 3 Layers by F1 Score:")
    print("-" * 60)
    top3 = df.nlargest(3, 'f1')[['layer', 'accuracy', 'precision', 'recall', 'f1']]
    print(top3.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, 
                       help='Input CSV file with layer sweep results')
    parser.add_argument('--output', type=str, default='layer_sweep_plot.png',
                       help='Output plot filename')
    args = parser.parse_args()
    
    plot_layer_sweep(args.input, args.output)