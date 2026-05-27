'''
Zero-shot CLIP classification helpers for all three CLIP variants using prompt ensembling.
'''

import torch
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)

# Move tokenized text to device.
def move_tokenized_to_device(tokenized, device):
    if hasattr(tokenized, 'to'):
        return tokenized.to(device)
    
    return tokenized

# Build text embeddings for each class.
def build_text_embeddings(model, tokenizer, prompt_registry, class_names, device='cuda'):
    text_embeddings = {}

    for class_name in class_names:
        prompts = prompt_registry[class_name]

        # Tokenize all prompts for this class.
        tokenized = tokenizer(prompts)
        tokenized = move_tokenized_to_device(tokenized, device)

        # Encode and normalize each prompt.
        with torch.no_grad():
            text_features = model.encode_text(tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Store all prompt-level embeddings for this class.
        text_embeddings[class_name] = text_features

    return text_embeddings

# Compute zero-shot predictions using prompt ensemble averaging.
def predict_from_image_features(image_features, text_embeddings, class_names, temperature=100.0):
    logits_by_class = []

    for class_name in class_names:
        class_text_embeddings = text_embeddings[class_name]

        # Compute similarity with each prompt for this class: [B, D] @ [D, P] = [B, P].
        prompt_logits = temperature * (image_features @ class_text_embeddings.T)

        # Average prompt-level logits for this class.
        class_logits = prompt_logits.mean(dim=1)

        logits_by_class.append(class_logits)

    # Stack class logits: [B, C].
    logits = torch.stack(logits_by_class, dim=1)

    # Convert to probabilities.
    probs = torch.softmax(logits, dim=1)

    # Get predictions.
    preds = logits.argmax(dim=1)

    return preds.cpu().numpy(), probs.cpu().numpy()

# End-to-end zero-shot prediction from images.
def zero_shot_predict(model, images, text_embeddings, class_names, device='cuda', temperature=100.0):
    images = images.to(device)

    with torch.no_grad():
        # Encode and normalize images.
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return predict_from_image_features(image_features, text_embeddings, class_names, temperature)

# Compute comprehensive classification metrics.
def compute_classification_metrics(true_labels, predictions, probabilities, class_names):
    metrics = {}
    labels = np.arange(len(class_names))
    
    # Overall metrics.
    metrics['accuracy'] = accuracy_score(true_labels, predictions)
    metrics['balanced_accuracy'] = balanced_accuracy_score(true_labels, predictions)
    metrics['macro_f1'] = f1_score(true_labels, predictions, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    
    # AUC.
    try:
        if len(class_names) == 2:
            metrics['auc'] = roc_auc_score(true_labels, probabilities[:, 1])
        else:
            metrics['auc'] = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='macro')
    except ValueError:
        metrics['auc'] = np.nan
    
    # Per-class metrics.
    precision_per_class = precision_score(true_labels, predictions, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predictions, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predictions, labels=labels, average=None, zero_division=0)
    
    for idx, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = precision_per_class[idx]
        metrics[f'{class_name}_recall'] = recall_per_class[idx]
        metrics[f'{class_name}_f1'] = f1_per_class[idx]
    
    return metrics

# Class for zero-shot CLIP evaluation.
class ZeroShotEvaluator:
    def __init__(self, model, preprocess, tokenizer, prompt_registry, class_names, device='cuda'):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.prompt_registry = prompt_registry
        self.class_names = class_names
        self.device = device
        self.text_embeddings = None
        
    def build_text_embeddings(self, verbose=True):
        '''Build text embeddings from prompt registry.'''
        self.text_embeddings = build_text_embeddings(
            self.model,
            self.tokenizer,
            self.prompt_registry,
            self.class_names,
            device=self.device
        )
        
        if verbose:
            for class_name in self.class_names:
                shape = self.text_embeddings[class_name].shape
                print(f'{class_name} - {shape}')
        
        return self.text_embeddings
    
    def encode_images(self, dataframe, batch_size=32, description='encoding images'):
        '''Encode images from dataframe.'''
        from .helpers import encode_images_batch
        
        image_features = encode_images_batch(
            self.model,
            self.preprocess,
            dataframe,
            device=self.device,
            batch_size=batch_size,
            description=description
        )
        return image_features
    
    def evaluate(self, dataframe, batch_size=32, temperature=100.0, description='encoding images'):
        '''Run full evaluation pipeline: encode images, predict, compute metrics.'''
        # Ensure text embeddings are built.
        if self.text_embeddings is None:
            self.build_text_embeddings(verbose=False)
        
        # Encode images.
        image_features = self.encode_images(dataframe, batch_size, description)
        
        # Get true labels.
        true_labels = dataframe['label_index'].values
        
        # Predict.
        predictions, probabilities = predict_from_image_features(
            image_features.to(self.device),
            self.text_embeddings,
            self.class_names,
            temperature=temperature
        )
        
        # Compute metrics.
        base_class_names = []
        for name in self.class_names:
            if ' tumor' in name:
                base_class_names.append(name.replace(' tumor', ''))
            elif ' scan' in name:
                base_class_names.append(name.replace(' scan', ''))
            else:
                base_class_names.append(name)
        
        metrics = compute_classification_metrics(
            true_labels,
            predictions,
            probabilities,
            base_class_names
        )
        
        return metrics, predictions, probabilities
    
    def print_results(self, metrics, model_name='CLIP'):
        '''Print evaluation results in a clean format.'''
        print(f'\nZero-Shot Results: {model_name}')
        print(f"accuracy - {metrics['accuracy']:.4f}")
        print(f"balanced Accuracy - {metrics['balanced_accuracy']:.4f}")
        print(f"macro F1 - {metrics['macro_f1']:.4f}")
        print(f"AUC - {metrics['auc']:.4f}")
        print(f"\nper-class F1 -")
        
        # Extract base class names from metrics keys.
        for key in metrics.keys():
            if key.endswith('_f1') and key not in ['macro_f1', 'weighted_f1']:
                class_name = key.replace('_f1', '')
                print(f"  {class_name}: {metrics[key]:.4f}")