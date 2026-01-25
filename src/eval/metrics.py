"""Evaluation metrics for vision-language tasks."""

import re
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


class VLMMetrics:
    """Metrics for Vision-Language Models."""
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        """Initialize metrics calculator.
        
        Args:
            tokenizer_name: Name of tokenizer for text processing.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
    
    def accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy.
        
        Args:
            predictions: List of predicted answers.
            references: List of reference answers.
            
        Returns:
            Accuracy score.
        """
        correct = sum(1 for pred, ref in zip(predictions, references) if pred.lower().strip() == ref.lower().strip())
        return correct / len(predictions) if predictions else 0.0
    
    def bleu_score(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Calculate BLEU scores.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference captions (can be multiple per prediction).
            
        Returns:
            Dictionary of BLEU scores.
        """
        bleu_scores = []
        
        for pred, refs in zip(predictions, references):
            # Tokenize
            pred_tokens = self._tokenize(pred)
            ref_tokens = [self._tokenize(ref) for ref in refs]
            
            # Calculate BLEU
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                smoothing_function=self.smoothing.method1,
            )
            bleu_scores.append(score)
        
        return {
            'bleu': np.mean(bleu_scores),
            'bleu_std': np.std(bleu_scores),
        }
    
    def rouge_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference captions.
            
        Returns:
            Dictionary of ROUGE scores.
        """
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL']),
        }
    
    def cider_score(self, predictions: List[str], references: List[List[str]]) -> float:
        """Calculate CIDEr score (simplified version).
        
        Args:
            predictions: List of predicted captions.
            references: List of reference captions.
            
        Returns:
            CIDEr score.
        """
        # Simplified CIDEr implementation
        # In practice, you'd want to use a proper CIDEr implementation
        cider_scores = []
        
        for pred, refs in zip(predictions, references):
            # Tokenize and create n-grams
            pred_tokens = self._tokenize(pred)
            ref_tokens_list = [self._tokenize(ref) for ref in refs]
            
            # Calculate n-gram precision for different n
            precisions = []
            for n in range(1, 5):  # 1-4 grams
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams_list = [self._get_ngrams(ref_tokens, n) for ref_tokens in ref_tokens_list]
                
                # Calculate precision
                precision = self._calculate_precision(pred_ngrams, ref_ngrams_list)
                precisions.append(precision)
            
            # Average precision
            cider_scores.append(np.mean(precisions))
        
        return np.mean(cider_scores)
    
    def retrieval_metrics(self, similarities: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Calculate retrieval metrics (R@K, MRR).
        
        Args:
            similarities: Similarity scores [batch_size, num_candidates].
            labels: Ground truth labels [batch_size].
            
        Returns:
            Dictionary of retrieval metrics.
        """
        batch_size = similarities.size(0)
        
        # Calculate ranks
        ranks = []
        for i in range(batch_size):
            sim_scores = similarities[i]
            label = labels[i].item()
            
            # Sort by similarity (descending)
            sorted_indices = torch.argsort(sim_scores, descending=True)
            rank = (sorted_indices == label).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        # Calculate metrics
        r1 = np.mean(ranks <= 1)
        r5 = np.mean(ranks <= 5)
        r10 = np.mean(ranks <= 10)
        mrr = np.mean(1.0 / ranks)
        
        return {
            'R@1': r1,
            'R@5': r5,
            'R@10': r10,
            'MRR': mrr,
        }
    
    def vqa_accuracy(self, predictions: List[str], references: List[List[str]]) -> float:
        """Calculate VQA accuracy (exact match with multiple acceptable answers).
        
        Args:
            predictions: List of predicted answers.
            references: List of reference answers (can be multiple per question).
            
        Returns:
            VQA accuracy.
        """
        correct = 0
        total = len(predictions)
        
        for pred, refs in zip(predictions, references):
            pred_clean = self._clean_answer(pred)
            refs_clean = [self._clean_answer(ref) for ref in refs]
            
            if pred_clean in refs_clean:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for evaluation.
        
        Args:
            text: Input text.
            
        Returns:
            List of tokens.
        """
        # Simple tokenization (in practice, use proper tokenizer)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Get n-grams from tokens.
        
        Args:
            tokens: List of tokens.
            n: N-gram size.
            
        Returns:
            List of n-grams.
        """
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _calculate_precision(self, pred_ngrams: List[str], ref_ngrams_list: List[List[str]]) -> float:
        """Calculate precision for n-grams.
        
        Args:
            pred_ngrams: Predicted n-grams.
            ref_ngrams_list: List of reference n-gram lists.
            
        Returns:
            Precision score.
        """
        if not pred_ngrams:
            return 0.0
        
        # Flatten all reference n-grams
        all_ref_ngrams = set()
        for ref_ngrams in ref_ngrams_list:
            all_ref_ngrams.update(ref_ngrams)
        
        # Calculate precision
        matches = sum(1 for ngram in pred_ngrams if ngram in all_ref_ngrams)
        return matches / len(pred_ngrams)
    
    def _clean_answer(self, answer: str) -> str:
        """Clean answer for VQA evaluation.
        
        Args:
            answer: Raw answer.
            
        Returns:
            Cleaned answer.
        """
        answer = answer.lower()
        answer = re.sub(r'[^\w\s]', '', answer)
        answer = ' '.join(answer.split())
        return answer


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: VLMMetrics,
    task: str = "vqa",
) -> Dict[str, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: Data loader for evaluation.
        device: Device to run evaluation on.
        metrics: Metrics calculator.
        task: Task type ('vqa', 'captioning', 'classification').
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_masks'].to(device)
            
            # Get model outputs
            if task == "vqa":
                outputs = model(input_ids=input_ids, pixel_values=images, attention_mask=attention_masks)
                # Generate answers (simplified)
                pred_answers = ["answer"] * images.size(0)  # Placeholder
                ref_answers = batch.get('answer', ['reference'] * images.size(0))
                
            elif task == "captioning":
                outputs = model(input_ids=input_ids, pixel_values=images, attention_mask=attention_masks)
                pred_answers = ["caption"] * images.size(0)  # Placeholder
                ref_answers = batch.get('caption', ['reference'] * images.size(0))
                
            elif task == "classification":
                outputs = model(input_ids=input_ids, pixel_values=images, attention_mask=attention_masks)
                pred_answers = ["class"] * images.size(0)  # Placeholder
                ref_answers = batch.get('label', [0] * images.size(0))
            
            predictions.extend(pred_answers)
            references.extend(ref_answers)
    
    # Calculate metrics
    if task == "vqa":
        return {
            'accuracy': metrics.vqa_accuracy(predictions, [[ref] for ref in references]),
        }
    elif task == "captioning":
        bleu_scores = metrics.bleu_score(predictions, [[ref] for ref in references])
        rouge_scores = metrics.rouge_score(predictions, references)
        cider_score = metrics.cider_score(predictions, [[ref] for ref in references])
        
        return {
            **bleu_scores,
            **rouge_scores,
            'cider': cider_score,
        }
    elif task == "classification":
        return {
            'accuracy': metrics.accuracy(predictions, [str(ref) for ref in references]),
        }
    
    return {}
