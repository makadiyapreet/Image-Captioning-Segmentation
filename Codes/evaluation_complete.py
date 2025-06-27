# src/utils/evaluation.py
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import math
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
from rouge_score import rouge_scorer
from tqdm import tqdm
import json
import os
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def calculate_bleu_score(references: List[List[List[str]]], 
                        hypotheses: List[List[str]], 
                        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)) -> float:
    """
    Calculate BLEU score for generated captions
    
    Args:
        references: List of reference captions (each image can have multiple references)
        hypotheses: List of generated captions
        weights: Weights for n-gram BLEU scores
    
    Returns:
        BLEU score
    """
    try:
        # Calculate corpus-level BLEU score
        bleu_score = corpus_bleu(references, hypotheses, weights=weights)
        return bleu_score
    except Exception as e:
        print(f"❌ Error calculating BLEU score: {e}")
        return 0.0

def calculate_individual_bleu_scores(references: List[List[List[str]]], 
                                   hypotheses: List[List[str]]) -> Dict[str, float]:
    """Calculate individual BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores"""
    scores = {}
    
    weights_dict = {
        'BLEU-1': (1.0, 0.0, 0.0, 0.0),
        'BLEU-2': (0.5, 0.5, 0.0, 0.0),
        'BLEU-3': (0.33, 0.33, 0.33, 0.0),
        'BLEU-4': (0.25, 0.25, 0.25, 0.25)
    }
    
    for name, weights in weights_dict.items():
        scores[name] = calculate_bleu_score(references, hypotheses, weights)
    
    return scores

def calculate_meteor_score(references: List[List[str]], 
                          hypotheses: List[str]) -> float:
    """
    Calculate METEOR score for generated captions
    
    Args:
        references: List of reference captions (first reference for each image)
        hypotheses: List of generated captions
    
    Returns:
        METEOR score
    """
    try:
        total_score = 0.0
        valid_scores = 0
        
        for ref_list, hyp in zip(references, hypotheses):
            if ref_list and hyp:
                # Use first reference for METEOR calculation
                ref = ref_list[0] if isinstance(ref_list[0], list) else ref_list
                ref_text = ' '.join(ref) if isinstance(ref, list) else ref
                hyp_text = ' '.join(hyp) if isinstance(hyp, list) else hyp
                
                score = meteor_score([ref_text], hyp_text)
                total_score += score
                valid_scores += 1
        
        return total_score / valid_scores if valid_scores > 0 else 0.0
    except Exception as e:
        print(f"❌ Error calculating METEOR score: {e}")
        return 0.0

def calculate_rouge_score(references: List[str], 
                         hypotheses: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE scores for generated captions
    
    Args:
        references: List of reference captions
        hypotheses: List of generated captions
    
    Returns:
        Dictionary with ROUGE scores
    """
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            if ref and hyp:
                scores = scorer.score(ref, hyp)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'ROUGE-1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
            'ROUGE-2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
            'ROUGE-L': np.mean(rougeL_scores) if rougeL_scores else 0.0
        }
    except Exception as e:
        print(f"❌ Error calculating ROUGE scores: {e}")
        return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}

def calculate_cider_score(references: List[List[str]], 
                         hypotheses: List[str]) -> float:
    """
    Calculate CIDEr score (simplified implementation)
    Note: This is a simplified version. For accurate CIDEr, use the official implementation.
    
    Args:
        references: List of reference captions for each image
        hypotheses: List of generated captions
    
    Returns:
        CIDEr score
    """
    try:
        def compute_tf(tokens):
            """Compute term frequency"""
            tf = Counter(tokens)
            return {token: count/len(tokens) for token, count in tf.items()}
        
        def compute_idf(all_tokens):
            """Compute inverse document frequency"""
            doc_freq = Counter()
            for tokens in all_tokens:
                doc_freq.update(set(tokens))
            
            total_docs = len(all_tokens)
            return {token: math.log(total_docs / freq) for token, freq in doc_freq.items()}
        
        def compute_tfidf(tokens, idf_dict):
            """Compute TF-IDF vector"""
            tf = compute_tf(tokens)
            tfidf = {}
            for token, tf_val in tf.items():
                tfidf[token] = tf_val * idf_dict.get(token, 0)
            return tfidf
        
        def cosine_similarity(vec1, vec2):
            """Compute cosine similarity between two vectors"""
            common_tokens = set(vec1.keys()) & set(vec2.keys())
            if not common_tokens:
                return 0.0
            
            dot_product = sum(vec1[token] * vec2[token] for token in common_tokens)
            norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
            norm2 = math.sqrt(sum(val**2 for val in vec2.values()))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        # Tokenize all sentences
        all_tokens = []
        ref_tokens = []
        hyp_tokens = []
        
        for ref_list, hyp in zip(references, hypotheses):
            # Process references
            ref_set = []
            for ref in ref_list:
                tokens = word_tokenize(ref.lower()) if isinstance(ref, str) else ref
                ref_set.append(tokens)
                all_tokens.append(tokens)
            ref_tokens.append(ref_set)
            
            # Process hypothesis
            hyp_tok = word_tokenize(hyp.lower()) if isinstance(hyp, str) else hyp
            hyp_tokens.append(hyp_tok)
            all_tokens.append(hyp_tok)
        
        # Compute IDF
        idf_dict = compute_idf(all_tokens)
        
        # Calculate CIDEr score
        cider_scores = []
        for ref_set, hyp_tok in zip(ref_tokens, hyp_tokens):
            hyp_tfidf = compute_tfidf(hyp_tok, idf_dict)
            
            # Average similarity with all references
            similarities = []
            for ref_tok in ref_set:
                ref_tfidf = compute_tfidf(ref_tok, idf_dict)
                sim = cosine_similarity(hyp_tfidf, ref_tfidf)
                similarities.append(sim)
            
            cider_scores.append(np.mean(similarities) if similarities else 0.0)
        
        return np.mean(cider_scores) if cider_scores else 0.0
    
    except Exception as e:
        print(f"❌ Error calculating CIDEr score: {e}")
        return 0.0

def calculate_bertscore(references: List[str], 
                       hypotheses: List[str],
                       model_type: str = "distilbert-base-uncased") -> Dict[str, float]:
    """
    Calculate BERTScore (requires bert-score package)
    
    Args:
        references: List of reference captions
        hypotheses: List of generated captions
        model_type: BERT model type to use
    
    Returns:
        Dictionary with BERTScore metrics
    """
    try:
        from bert_score import score
        
        P, R, F1 = score(hypotheses, references, model_type=model_type, verbose=False)
        
        return {
            'BERTScore-P': P.mean().item(),
            'BERTScore-R': R.mean().item(),
            'BERTScore-F1': F1.mean().item()
        }
    except ImportError:
        print("❌ bert-score package not installed. Install with: pip install bert-score")
        return {'BERTScore-P': 0.0, 'BERTScore-R': 0.0, 'BERTScore-F1': 0.0}
    except Exception as e:
        print(f"❌ Error calculating BERTScore: {e}")
        return {'BERTScore-P': 0.0, 'BERTScore-R': 0.0, 'BERTScore-F1': 0.0}

def calculate_diversity_metrics(hypotheses: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for generated captions
    
    Args:
        hypotheses: List of generated captions
    
    Returns:
        Dictionary with diversity metrics
    """
    try:
        # Tokenize all hypotheses
        all_tokens = []
        for hyp in hypotheses:
            tokens = word_tokenize(hyp.lower()) if isinstance(hyp, str) else hyp
            all_tokens.extend(tokens)
        
        # Calculate unique tokens
        unique_tokens = set(all_tokens)
        total_tokens = len(all_tokens)
        
        # Type-Token Ratio
        ttr = len(unique_tokens) / total_tokens if total_tokens > 0 else 0.0
        
        # Calculate n-gram diversity
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        # Unigram diversity
        unigrams = [token for hyp in hypotheses for token in 
                   (word_tokenize(hyp.lower()) if isinstance(hyp, str) else hyp)]
        unique_unigrams = len(set(unigrams))
        total_unigrams = len(unigrams)
        unigram_diversity = unique_unigrams / total_unigrams if total_unigrams > 0 else 0.0
        
        # Bigram diversity
        all_bigrams = []
        for hyp in hypotheses:
            tokens = word_tokenize(hyp.lower()) if isinstance(hyp, str) else hyp
            bigrams = get_ngrams(tokens, 2)
            all_bigrams.extend(bigrams)
        
        unique_bigrams = len(set(all_bigrams))
        total_bigrams = len(all_bigrams)
        bigram_diversity = unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0
        
        # Average sentence length
        avg_length = np.mean([len(word_tokenize(hyp.lower()) if isinstance(hyp, str) else hyp) 
                             for hyp in hypotheses])
        
        return {
            'TTR': ttr,
            'Unigram_Diversity': unigram_diversity,
            'Bigram_Diversity': bigram_diversity,
            'Avg_Length': avg_length,
            'Unique_Words': unique_tokens.__len__(),
            'Total_Words': total_tokens
        }
    except Exception as e:
        print(f"❌ Error calculating diversity metrics: {e}")
        return {
            'TTR': 0.0,
            'Unigram_Diversity': 0.0,
            'Bigram_Diversity': 0.0,
            'Avg_Length': 0.0,
            'Unique_Words': 0,
            'Total_Words': 0
        }

def preprocess_caption(caption: str) -> str:
    """
    Preprocess caption text for evaluation
    
    Args:
        caption: Input caption string
    
    Returns:
        Preprocessed caption
    """
    if not isinstance(caption, str):
        return str(caption)
    
    # Convert to lowercase
    caption = caption.lower()
    
    # Remove extra whitespace
    caption = re.sub(r'\s+', ' ', caption)
    
    # Remove punctuation at the end
    caption = re.sub(r'[.!?]+$', '', caption)
    
    # Strip whitespace
    caption = caption.strip()
    
    return caption

def evaluate_captions(references: List[List[str]], 
                     hypotheses: List[str],
                     include_bertscore: bool = False,
                     include_diversity: bool = True,
                     verbose: bool = True) -> Dict[str, float]:
    """
    Comprehensive evaluation of generated captions
    
    Args:
        references: List of reference captions for each image
        hypotheses: List of generated captions
        include_bertscore: Whether to include BERTScore calculation
        include_diversity: Whether to include diversity metrics
        verbose: Whether to print progress
    
    Returns:
        Dictionary with all evaluation metrics
    """
    if verbose:
        print("🔍 Starting comprehensive caption evaluation...")
    
    # Preprocess captions
    processed_hyp = [preprocess_caption(hyp) for hyp in hypotheses]
    processed_ref = []
    
    for ref_list in references:
        processed_ref_list = [preprocess_caption(ref) for ref in ref_list]
        processed_ref.append(processed_ref_list)
    
    results = {}
    
    # Prepare data for different metrics
    # For BLEU: tokenized format
    ref_tokens = [[word_tokenize(ref) for ref in ref_list] for ref_list in processed_ref]
    hyp_tokens = [word_tokenize(hyp) for hyp in processed_hyp]
    
    # For other metrics: first reference only
    ref_first = [ref_list[0] for ref_list in processed_ref]
    
    # Calculate BLEU scores
    if verbose:
        print("📊 Calculating BLEU scores...")
    bleu_scores = calculate_individual_bleu_scores(ref_tokens, hyp_tokens)
    results.update(bleu_scores)
    
    # Calculate METEOR score
    if verbose:
        print("🌠 Calculating METEOR score...")
    meteor = calculate_meteor_score(processed_ref, processed_hyp)
    results['METEOR'] = meteor
    
    # Calculate ROUGE scores
    if verbose:
        print("🔴 Calculating ROUGE scores...")
    rouge_scores = calculate_rouge_score(ref_first, processed_hyp)
    results.update(rouge_scores)
    
    # Calculate CIDEr score
    if verbose:
        print("🎯 Calculating CIDEr score...")
    cider = calculate_cider_score(processed_ref, processed_hyp)
    results['CIDEr'] = cider
    
    # Calculate BERTScore (optional)
    if include_bertscore:
        if verbose:
            print("🤖 Calculating BERTScore...")
        bert_scores = calculate_bertscore(ref_first, processed_hyp)
        results.update(bert_scores)
    
    # Calculate diversity metrics (optional)
    if include_diversity:
        if verbose:
            print("🌈 Calculating diversity metrics...")
        diversity_scores = calculate_diversity_metrics(processed_hyp)
        results.update(diversity_scores)
    
    if verbose:
        print("✅ Evaluation completed!")
    
    return results

def print_evaluation_results(results: Dict[str, float], 
                           title: str = "Evaluation Results"):
    """
    Pretty print evaluation results
    
    Args:
        results: Dictionary with evaluation metrics
        title: Title for the results
    """
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    
    # Group metrics by type
    bleu_metrics = {k: v for k, v in results.items() if 'BLEU' in k}
    rouge_metrics = {k: v for k, v in results.items() if 'ROUGE' in k}
    bert_metrics = {k: v for k, v in results.items() if 'BERT' in k}
    diversity_metrics = {k: v for k, v in results.items() if k in 
                        ['TTR', 'Unigram_Diversity', 'Bigram_Diversity', 'Avg_Length', 
                         'Unique_Words', 'Total_Words']}
    other_metrics = {k: v for k, v in results.items() if k not in 
                    list(bleu_metrics.keys()) + list(rouge_metrics.keys()) + 
                    list(bert_metrics.keys()) + list(diversity_metrics.keys())}
    
    # Print each group
    if bleu_metrics:
        print("\n📈 BLEU Scores:")
        for metric, score in bleu_metrics.items():
            print(f"  {metric:<12}: {score:.4f}")
    
    if other_metrics:
        print("\n🎯 Other Metrics:")
        for metric, score in other_metrics.items():
            if metric in ['METEOR', 'CIDEr']:
                print(f"  {metric:<12}: {score:.4f}")
    
    if rouge_metrics:
        print("\n🔴 ROUGE Scores:")
        for metric, score in rouge_metrics.items():
            print(f"  {metric:<12}: {score:.4f}")
    
    if bert_metrics:
        print("\n🤖 BERTScore:")
        for metric, score in bert_metrics.items():
            print(f"  {metric:<12}: {score:.4f}")
    
    if diversity_metrics:
        print("\n🌈 Diversity Metrics:")
        for metric, score in diversity_metrics.items():
            if isinstance(score, int):
                print(f"  {metric:<16}: {score}")
            else:
                print(f"  {metric:<16}: {score:.4f}")
    
    print(f"{'='*50}\n")

def save_evaluation_results(results: Dict[str, float], 
                          filepath: str):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Dictionary with evaluation metrics
        filepath: Path to save the results
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

# Example usage
if __name__ == "__main__":
    # Example data
    references = [
        ["A cat sitting on a mat", "A feline resting on a rug"],
        ["A dog running in the park", "A canine playing outdoors"],
        ["A bird flying in the sky", "A bird soaring through the air"]
    ]
    
    hypotheses = [
        "A cat on a mat",
        "A dog in the park",
        "A bird in the sky"
    ]
    
    # Run evaluation
    results = evaluate_captions(references, hypotheses, include_bertscore=False)
    print_evaluation_results(results)
    
    # Save results
    save_evaluation_results(results, "evaluation_results.json")