import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from .models import (load_openai_clip, load_biomedclip, load_unimedclip, make_openai_clip_loader, make_biomedclip_loader, make_unimedclip_loader)


# Load BUSI grayscale image and prepare for CLIP.
def load_busi_image_as_pil(image_path, output_size=224):
    # Load as grayscale PIL image.
    image = Image.open(image_path).convert('L')
    # Resize with aspect-ratio preservation and zero-padding.
    image = ImageOps.pad(image, (output_size, output_size), color=0, centering=(0.5, 0.5))
    # Convert grayscale to RGB.
    image = image.convert('RGB')
    return image

# Encode images in batches with CLIP model.
def encode_images_batch(model, preprocess, dataframe, device='cuda', batch_size=32, description='encoding images'):
    all_features = []
    image_paths = dataframe['image_path'].tolist()
    
    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc=description):
            batch_paths = image_paths[start:start + batch_size]
            
            # Load and preprocess images.
            pil_images = [load_busi_image_as_pil(p, output_size=224) for p in batch_paths]
            images = torch.stack([preprocess(img) for img in pil_images]).to(device)
            
            # Encode images.
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            
            all_features.append(features.cpu())
    
    # Concatenate all features.
    return torch.cat(all_features, dim=0)

# Load BUSI train/val/test splits.
def load_busi_splits(project_root, busi_classes):
    split_directory = Path(project_root) / 'dataset' / 'split'
    label_to_index = {cls: idx for idx, cls in enumerate(busi_classes)}
    
    splits = {}
    for split_name in ['train', 'val', 'test']:
        image_data = []
        for category in busi_classes:
            category_directory = split_directory / split_name / category
            if category_directory.exists():
                for image_path in category_directory.glob('*.png'):
                    if '_mask' not in image_path.name:
                        image_data.append({
                            'image_path': str(image_path),
                            'label': category,
                            'label_index': label_to_index[category]
                        })
        
        splits[split_name] = pd.DataFrame(image_data)
    
    return splits['train'], splits['val'], splits['test']

# Save results to CSV.
def save_results_to_csv(results_dict, results_path, model_name, append=False):
    results_df = pd.DataFrame([{
        'model': model_name,
        **results_dict
    }])
    
    if append and Path(results_path).exists():
        existing = pd.read_csv(results_path)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    
    results_df.to_csv(results_path, index=False)
    return results_df


def plot_fewshot_summary(summary_df, metric='test_macro_f1', output_path=None, title=None):
    """Plot few-shot summary statistics for a single experiment type.

    The plot shows the mean metric value across seeds for each shot count,
    with error shading representing one standard deviation.
    """
    if summary_df is None or summary_df.empty:
        raise ValueError('summary_df must be a non-empty DataFrame')

    metric_mean = f'{metric}_mean'
    metric_std = f'{metric}_std'

    if metric_mean not in summary_df.columns or metric_std not in summary_df.columns:
        raise ValueError(f'summary_df must contain columns {metric_mean} and {metric_std}')

    x = summary_df['k'].astype(int).values
    y = summary_df[metric_mean].values
    y_err = summary_df[metric_std].fillna(0).values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker='o', linewidth=2)
    ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
    ax.set_xlabel('shots per class (k)')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'Few-shot {metric.replace("_", " ").title()}')
    ax.grid(True, linestyle='--', alpha=0.4)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close(fig)
    return fig


def plot_fewshot_comparison(lp_summary_df, lora_summary_df, metric='test_macro_f1', output_path=None, title=None):
    """Compare LP and LoRA few-shot performance curves on the same metric.

    This plot is useful for direct baseline vs. adapter comparison when both
    summaries are available in the same experiment script.
    """
    if lp_summary_df is None or lp_summary_df.empty:
        raise ValueError('lp_summary_df must be a non-empty DataFrame')
    if lora_summary_df is None or lora_summary_df.empty:
        raise ValueError('lora_summary_df must be a non-empty DataFrame')

    metric_mean = f'{metric}_mean'

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lp_summary_df['k'], lp_summary_df[metric_mean], marker='o', label='Linear Probe')
    ax.plot(lora_summary_df['k'], lora_summary_df[metric_mean], marker='s', label='LoRA')
    ax.set_xlabel('shots per class (k)')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'Linear Probe vs LoRA: {metric.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close(fig)
    return fig

# Single-layer classifier for both LP and LoRA experiments. We might need to experiment with feature reduction and feature processing blocks for the classification head (e.g. TimmCLIPAdapter).
class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classification_head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classification_head(x)