import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import torch
from rich.console import Console
from rich.table import Table
from rich import box
import evaluate

# Load metrics
try:
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    bertscore = evaluate.load('bertscore') 
except Exception as e:
    print(f"[bold red]Critical Error:[/bold red] Could not load metrics. Please run: uv pip install evaluate rouge_score bert_score absl-py sacrebleu")
    print(f"Error details: {e}")
    exit(1)

def calculate_metrics(predictions, references, bertscore_model="distilbert-base-uncased",
                      bertscore_num_layers=5):
    """
    Calculates ROUGE, BLEU, and BERTScore for a list of predictions vs references.
    """
    results = {}
    
    # ROUGE
    # The evaluate wrapper's default BootstrapAggregator samples with NumPy randomness
    # and returns the median of bootstrap means. Identical predictions therefore produced
    # three different ROUGE aggregates in three bounded battery runs. Compute each row's
    # F1 and take the direct arithmetic mean instead.
    rouge_res = rouge.compute(
        predictions=predictions, references=references, use_aggregator=False)
    for name in ('rouge1', 'rouge2', 'rougeL'):
        values = rouge_res[name]
        results[name] = sum(values) / len(values)
    
    # BLEU
    bleu_res = bleu.compute(predictions=predictions, references=references)
    results['bleu'] = bleu_res['bleu']

    # BERTScore (optional, can be slow)
    # The fixed battery passes an immutable local snapshot here. `num_layers` must be
    # explicit for a local path because bert-score's lookup table is keyed by the mutable
    # Hub model name, not by local directories.
    results_bert = bertscore.compute(
        predictions=predictions, references=references, lang="en",
        model_type=str(bertscore_model), num_layers=bertscore_num_layers)
    results['bertscore_f1'] = sum(results_bert['f1']) / len(results_bert['f1'])

    return results


def metric_runtime_identity(bertscore_model="distilbert-base-uncased",
                            bertscore_num_layers=5):
    """Record the evaluator implementations that produced an aggregate."""
    packages = {}
    for package in ("evaluate", "rouge-score", "bert-score", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    model_path = Path(bertscore_model)
    model_files = None
    if model_path.is_dir():
        model_files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(model_path.iterdir()) if path.is_file()
        }
    return {
        "packages": packages,
        "evaluate_module_hashes": {
            "rouge": getattr(rouge, "_hash", None),
            "bleu": getattr(bleu, "_hash", None),
            "bertscore": getattr(bertscore, "_hash", None),
        },
        "rouge_aggregation": "arithmetic mean of per-row F1; use_aggregator=False",
        "bertscore_model": str(bertscore_model),
        "bertscore_num_layers": bertscore_num_layers,
        "bertscore_model_files_sha256": model_files,
    }

def main():
    parser = argparse.ArgumentParser(description="Run evaluations on generated JSON.")
    parser.add_argument("--input_json", type=str, default="inference_generations.json", help="Path to generations JSON")
    parser.add_argument("--output_results", type=str, default="eval_results.json", help="Path to save evaluation scores")
    args = parser.parse_args()
    
    console = Console()
    
    # Load Data
    with open(args.input_json, 'r') as f:
        data = json.load(f)
        
    if not data:
        console.print("[bold red]No data found in input JSON.[/bold red]")
        return

    # Identify Models
    # Look at the first sample to find model keys
    model_names = list(data[0]["predictions"].keys())
    console.print(f"Found generations for models: [cyan]{', '.join(model_names)}[/cyan]")
    
    # Organize data per model
    model_data = {name: {"preds": [], "refs": []} for name in model_names}
    
    for item in data:
        ground_truth = item["ground_truth"]
        for name in model_names:
            if name in item["predictions"]:
                pred = item["predictions"][name]
                model_data[name]["preds"].append(pred)
                model_data[name]["refs"].append(ground_truth)
    
    # Run Evaluations
    console.rule("[bold blue]Running Evaluations")
    
    final_results = {
        "aggregate": {},
        "instances": []
    }
    
    table = Table(title="Evaluation Results", box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("ROUGE-1", style="green")
    table.add_column("ROUGE-L", style="green")
    table.add_column("DistilBERT F1", style="magenta")
    
    # Calculate aggregate scores
    for model_name in model_names:
        preds = model_data[model_name]["preds"]
        refs = model_data[model_name]["refs"]
        
        if not preds:
            continue
            
        console.print(f"Evaluating {model_name}...")
        scores = calculate_metrics(preds, refs)
        final_results["aggregate"][model_name] = scores
        
        table.add_row(
            model_name,
            f"{scores['rouge1']:.4f}",
            f"{scores['rougeL']:.4f}",
            f"{scores.get('bertscore_f1', 0):.4f}"
        )
        
    console.print(table)
    
    # Calculate per-instance scores for detailed viewing
    console.print("\n[yellow]Calculating per-instance metrics...[/yellow]")
    for item in data:
        instance_entry = {
            "id": item.get("id"),
            "prefix": item["prefix"],
            "ground_truth": item["ground_truth"],
            "predictions": item["predictions"],
            "metrics": {}
        }
        
        ref = item["ground_truth"]
        for model_name, pred in item["predictions"].items():
            # Instance metrics (ROUGE/BLEU only for speed)
            inst_scores = {}
            r_res = rouge.compute(predictions=[pred], references=[ref])
            inst_scores['rouge1'] = r_res['rouge1']
            inst_scores['rougeL'] = r_res['rougeL']
            
            b_res = bleu.compute(predictions=[pred], references=[ref])
            inst_scores['bleu'] = b_res['bleu']
            
            instance_entry["metrics"][model_name] = inst_scores
            
        final_results["instances"].append(instance_entry)

    # Save results
    with open(args.output_results, 'w') as f:
        json.dump(final_results, f, indent=4)
        
    console.print(f"\n[bold green]Evaluation results saved to {args.output_results}[/bold green]")

if __name__ == "__main__":
    main()
