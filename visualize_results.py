import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime

def load_result(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_retriever_comparison(data, output_dir, timestamp):
    retriever_data = data.get('retriever_evaluation', {})
    if not retriever_data:
        print("No retriever evaluation data found.")
        return

    # Prepare data for plotting
    plot_df = []
    for r_type, metrics in retriever_data.items():
        for metric, value in metrics.items():
            plot_df.append({
                'Retriever': r_type.capitalize(),
                'Metric': metric.upper(),
                'Value': value
            })
    
    df = pd.DataFrame(plot_df)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x='Metric', y='Value', hue='Retriever', data=df, palette='viridis')
    
    plt.title(f'Retriever Performance Comparison\n({timestamp})', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)
    
    output_path = os.path.join(output_dir, f'retriever_comparison_{timestamp.replace(":", "-").replace(".", "-")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Retriever comparison plot saved to: {output_path}")

def plot_generation_metrics(data, output_dir, timestamp):
    gen_eval = data.get('generation_evaluation', {})
    ragas_results = gen_eval.get('ragas_results', {})
    
    if not ragas_results:
        print("No Ragas results found.")
        return

    # Prepare data for boxplot
    df_list = []
    for metric, values in ragas_results.items():
        for val in values:
            df_list.append({'Metric': metric.replace('_', ' ').capitalize(), 'Score': val})
    
    df = pd.DataFrame(df_list)
    
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # Boxplot shows distribution
    sns.boxplot(x='Metric', y='Score', data=df, palette='Set2')
    # Stripplot shows individual points
    sns.stripplot(x='Metric', y='Score', data=df, color=".25", alpha=0.5)
    
    plt.title(f'Generation Metrics Distribution (Ragas)\n({timestamp})', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(-0.05, 1.05)
    
    output_path = os.path.join(output_dir, f'generation_metrics_{timestamp.replace(":", "-").replace(".", "-")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generation metrics plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize RAG evaluation results.')
    parser.add_argument('--file', type=str, help='Path to the evaluation JSON file.')
    parser.add_argument('--output_dir', type=str, default='analysis', help='Directory to save plots.')
    
    args = parser.parse_args()
    
    if not args.file:
        # Try to find the latest result file
        results_dir = 'results'
        files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not files:
            print("No JSON result files found in 'results/' directory.")
            return
        files.sort()
        args.file = os.path.join(results_dir, files[-1])
        print(f"No file specified. Using latest: {args.file}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")

    data = load_result(args.file)
    timestamp = data.get('timestamp', 'unknown_time')
    
    plot_retriever_comparison(data, args.output_dir, timestamp)
    plot_generation_metrics(data, args.output_dir, timestamp)

if __name__ == "__main__":
    main()
