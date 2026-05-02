#!/usr/bin/env python3
# Licensed under the MIT License.

"""
SSDA Multi-Target Evaluation Script
Automatically runs SSDA algorithms (S+T, MME, and ENT) across all valid target domains
and computes mean accuracy following the SSDA paper protocol.
"""

import argparse
import copy
import logging
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from semilearn.algorithms import get_algorithm, name2alg
from semilearn.core.utils import (
    TBLog,
    count_parameters,
    get_logger,
    get_net_builder,
    get_port,
    over_write_args_from_file,
    send_model_cuda,
)
from semilearn.imb_algorithms import get_imb_algorithm, name2imbalg


def detect_source_domain(pretrain_path):
    """Detect source domain from pretrained model path"""
    if 'photo' in pretrain_path: return 'photo'
    elif 'art_painting' in pretrain_path: return 'art_painting'
    elif 'cartoon' in pretrain_path: return 'cartoon'
    elif 'sketch' in pretrain_path: return 'sketch'
    else: raise ValueError(f'Cannot detect source from pretrain_path: {pretrain_path}')


def get_target_domains(dataset, source_domain):
    """Get list of valid target domains for given dataset and source"""
    domains_map = {
        'acs': ['art_painting','cartoon','sketch'],
        'pcs': ['photo','cartoon','sketch'],
        'pas': ['photo','art_painting','sketch'],
        'pac': ['photo','art_painting','cartoon'],
    }
    valid_targets = domains_map[dataset]
    return [d for d in valid_targets if d != source_domain]


def train_single_target(base_args, target_domain, logger=None):
    """Train and evaluate on a single target domain"""
    # Create a deep copy of args for this target
    args = copy.deepcopy(base_args)
    args.target_domain = target_domain
    
    # Update save_name to include target domain
    base_save_name = args.save_name.replace('_photo', '').replace('_art_painting', '').replace('_cartoon', '').replace('_sketch', '')
    args.save_name = f"{base_save_name}_{target_domain}"
    
    # Create save path
    save_path = os.path.join(args.save_dir, args.save_name)
    
    # Setup logging for this target
    if logger is None:
        logger = get_logger(args.save_name, save_path, "INFO")
    
    logger.info(f"=" * 60)
    logger.info(f"Training {args.algorithm.upper()} on target domain: {target_domain}")
    
    # Handle case when pretrain_path is not set
    if hasattr(args, 'pretrain_path') and args.pretrain_path:
        logger.info(f"Source domain: {detect_source_domain(args.pretrain_path)}")
    else:
        logger.info(f"Source domain: Detected from algorithm implementation")
    
    logger.info(f"Save path: {save_path}")
    logger.info(f"=" * 60)
    
    try:
        # Setup tensorboard
        tb_log = TBLog(save_path, "tensorboard", use_tensorboard=args.use_tensorboard)
        
        # Build network (fresh for each target)
        _net_builder = get_net_builder(args.net, args.net_from_name)
        
        # Create algorithm instance
        if args.imb_algorithm is not None:
            model = get_imb_algorithm(args, _net_builder, tb_log, logger)
        else:
            model = get_algorithm(args, _net_builder, tb_log, logger)
            
        logger.info(f"Number of Trainable Params: {count_parameters(model.model)}")
        
        # Send model to GPU
        model.model = send_model_cuda(args, model.model)
        if hasattr(model, 'ema_model') and model.ema_model is not None:
            model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
        
        # Train the model
        logger.info("Starting training...")
        model.train()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = model.evaluate('test')
        test_acc = test_results.get('test/top-1-acc', 0.0)
        
        logger.info(f"Target {target_domain} - Test accuracy: {test_acc:.4f}")
        
        # Log all results
        for key, item in model.results_dict.items():
            logger.info(f"Model result - {key} : {item}")
            
        return target_domain, test_acc, model.results_dict
        
    except Exception as e:
        logger.error(f"Error training on target {target_domain}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return target_domain, 0.0, {}


def main_multi_target(args):
    """Main function to run multi-target evaluation"""
    
    # Setup random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    # Detect source and get target domains
    if hasattr(args, 'pretrain_path') and args.pretrain_path:
        source_domain = detect_source_domain(args.pretrain_path)
    else:
        # Fallback: use first domain from dataset as source
        domains_map = {
            'acs': ['art_painting','cartoon','sketch'],
            'pcs': ['photo','cartoon','sketch'], 
            'pas': ['photo','art_painting','sketch'],
            'pac': ['photo','art_painting','cartoon'],
        }
        all_domains = ['photo'] + domains_map[args.dataset]
        source_domain = all_domains[0]  # Default to photo
    
    target_domains = get_target_domains(args.dataset, source_domain)
    
    # Setup main logger
    main_save_path = os.path.join(args.save_dir, f"{args.algorithm}_{args.dataset}_multi_target")
    os.makedirs(main_save_path, exist_ok=True)
    main_logger = get_logger(f"{args.algorithm}_multi_target", main_save_path, "INFO")
    
    main_logger.info(f"Starting multi-target evaluation for {args.algorithm.upper()}")
    main_logger.info(f"Dataset: {args.dataset}")
    main_logger.info(f"Source domain: {source_domain}")
    main_logger.info(f"Target domains: {target_domains}")
    
    # Handle pretrain_path logging
    if hasattr(args, 'pretrain_path') and args.pretrain_path:
        main_logger.info(f"Pretrained model: {args.pretrain_path}")
    else:
        main_logger.info(f"Pretrained model: None (training from scratch)")
    
    # Run training for each target domain
    results = []
    for i, target_domain in enumerate(target_domains, 1):
        main_logger.info(f"\n[{i}/{len(target_domains)}] Processing target domain: {target_domain}")
        
        target_name, test_acc, target_results = train_single_target(args, target_domain, main_logger)
        results.append((target_name, test_acc, target_results))
        
        main_logger.info(f"Completed {target_name}: {test_acc:.4f}")
    
    # Compute and report mean accuracy
    test_accs = [acc for _, acc, _ in results]
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    
    main_logger.info(f"\n" + "="*80)
    main_logger.info(f"FINAL RESULTS - {args.algorithm.upper()} on {args.dataset.upper()}")
    main_logger.info(f"Source domain: {source_domain}")
    main_logger.info(f"="*80)
    
    for target_name, test_acc, _ in results:
        main_logger.info(f"{target_name:>15}: {test_acc:.4f}")
    
    main_logger.info(f"{'='*50}")
    main_logger.info(f"{'Mean':>15}: {mean_acc:.4f} ± {std_acc:.4f}")
    main_logger.info(f"="*80)
    
    # Save results to file
    results_file = os.path.join(main_save_path, "multi_target_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Multi-target evaluation results\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Source: {source_domain}\n")
        f.write(f"Targets: {target_domains}\n")
        
        # Handle pretrain_path in results file
        if hasattr(args, 'pretrain_path') and args.pretrain_path:
            f.write(f"Pretrained: {args.pretrain_path}\n\n")
        else:
            f.write(f"Pretrained: None (from scratch)\n\n")
        
        f.write("Per-target results:\n")
        for target_name, test_acc, _ in results:
            f.write(f"  {target_name}: {test_acc:.4f}\n")
        
        f.write(f"\nMean: {mean_acc:.4f} ± {std_acc:.4f}\n")
    
    main_logger.info(f"Results saved to: {results_file}")
    return mean_acc, results


def get_config():
    """Parse command line arguments"""
    from semilearn.algorithms.utils import str2bool
    
    parser = argparse.ArgumentParser(description="SSDA Multi-Target Evaluation")
    
    # Config file
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    
    # Override specific parameters if needed
    parser.add_argument('--algorithm', type=str, choices=['ssda_st', 'ssda_mme', 'ssda_ent'],
                       help='Algorithm to run (overrides config)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU to use')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load config from file
    over_write_args_from_file(args, args.config)
    
    # Set default values for attributes that may not be in config but are required
    default_attrs = {
        'imb_algorithm': None,
        'distributed': False,
        'multiprocessing_distributed': False,
        'world_size': 1,
        'rank': 0,
        'dist_backend': 'nccl',
        'dist_url': 'tcp://127.0.0.1:10030',
        'use_aim': False,
        'clip': 0.0,
        'clip_grad': 0,
        'use_cat': True,
        'use_pretrain': True,
        'net_from_name': False,
        'layer_decay': 0.5,
        'hard_label': True,
        'include_lb_to_ulb': True,
        'lb_imb_ratio': 1,
        'ulb_imb_ratio': 1,
        'ulb_num_labels': None,
        'sample_rate': 16000,
        'max_length': 512,
        'max_length_seconds': 4.0,
    }
    
    for attr, default_val in default_attrs.items():
        if not hasattr(args, attr):
            setattr(args, attr, default_val)
    
    return args


if __name__ == "__main__":
    # Parse arguments
    args = get_config()
    
    # Ensure single GPU mode for simplicity
    args.distributed = False
    args.multiprocessing_distributed = False
    args.world_size = 1
    args.rank = 0
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {args.gpu}")
    else:
        print("CUDA not available, using CPU")
        args.gpu = None
    
    # Run multi-target evaluation
    mean_acc, results = main_multi_target(args)
    
    print(f"\nFinal mean accuracy: {mean_acc:.4f}")
