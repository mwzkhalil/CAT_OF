import math
import time
import torch
import wandb
import numpy
import random
import argparse
import contextlib
import json
import os
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.collators import VQACollator, MMStarCollator
from data.datasets import MMStarDataset, VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import CAT
import models.config as config
import models.utils as utils

#Otherwise, the tokenizer will through a warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TimeComplexityTracker:
    """Comprehensive time complexity and performance tracker"""
    def __init__(self):
        self.training_times = []
        self.testing_times = []
        self.batch_times = []
        self.epoch_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        self.data_loading_times = []
        self.memory_usage = []
        self.throughput_data = []
        self.accuracy_history = []
        self.loss_history = []
        
    def start_timer(self, name):
        setattr(self, f"{name}_start", time.time())
        
    def stop_timer(self, name):
        start_time = getattr(self, f"{name}_start", time.time())
        duration = time.time() - start_time
        getattr(self, f"{name}_times", []).append(duration)
        return duration
        
    def log_memory_usage(self):
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.memory_usage.append(memory_mb)
            return memory_mb
        return 0
        
    def log_throughput(self, batch_size, tokens_per_second, samples_per_second):
        self.throughput_data.append({
            'batch_size': batch_size,
            'tokens_per_second': tokens_per_second,
            'samples_per_second': samples_per_second,
            'timestamp': time.time()
        })
        
    def analyze_complexity(self):
        """Analyze time complexity patterns"""
        analysis = {
            'avg_training_time': np.mean(self.training_times) if self.training_times else 0,
            'avg_testing_time': np.mean(self.testing_times) if self.testing_times else 0,
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'avg_forward_time': np.mean(self.forward_times) if self.forward_times else 0,
            'avg_backward_time': np.mean(self.backward_times) if self.backward_times else 0,
            'avg_optimizer_time': np.mean(self.optimizer_times) if self.optimizer_times else 0,
            'peak_memory_mb': max(self.memory_usage) if self.memory_usage else 0,
            'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'total_training_time': sum(self.training_times),
            'training_efficiency': len(self.batch_times) / sum(self.batch_times) if sum(self.batch_times) > 0 else 0
        }
        return analysis
        
    def plot_performance_metrics(self, save_path="performance_analysis.png"):
        """Generate comprehensive performance plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training time over epochs
        if self.epoch_times:
            axes[0,0].plot(self.epoch_times)
            axes[0,0].set_title('Training Time per Epoch')
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Time (seconds)')
            
        # Memory usage over time
        if self.memory_usage:
            axes[0,1].plot(self.memory_usage)
            axes[0,1].set_title('GPU Memory Usage')
            axes[0,1].set_xlabel('Step')
            axes[0,1].set_ylabel('Memory (MB)')
            
        # Throughput analysis
        if self.throughput_data:
            tokens_per_sec = [d['tokens_per_second'] for d in self.throughput_data]
            axes[0,2].plot(tokens_per_sec)
            axes[0,2].set_title('Tokens per Second')
            axes[0,2].set_xlabel('Step')
            axes[0,2].set_ylabel('Tokens/sec')
            
        # Batch processing time distribution
        if self.batch_times:
            axes[1,0].hist(self.batch_times, bins=50, alpha=0.7)
            axes[1,0].set_title('Batch Processing Time Distribution')
            axes[1,0].set_xlabel('Time (seconds)')
            axes[1,0].set_ylabel('Frequency')
            
        # Forward vs Backward time comparison
        if self.forward_times and self.backward_times:
            min_len = min(len(self.forward_times), len(self.backward_times))
            axes[1,1].scatter(self.forward_times[:min_len], self.backward_times[:min_len], alpha=0.6)
            axes[1,1].set_title('Forward vs Backward Time')
            axes[1,1].set_xlabel('Forward Time (s)')
            axes[1,1].set_ylabel('Backward Time (s)')
            
        # Accuracy over time
        if self.accuracy_history:
            axes[1,2].plot(self.accuracy_history)
            axes[1,2].set_title('Accuracy Over Training')
            axes[1,2].set_xlabel('Evaluation Step')
            axes[1,2].set_ylabel('Accuracy')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance analysis saved to {save_path}")

def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def destroy_dist():
    dist.destroy_process_group()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank() == 0 if is_dist() else True

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all

def wrap_model(model):
    return DistributedDataParallel(model, device_ids=[dist.get_rank()])

def get_run_name(train_cfg):
    dataset_size = "full_ds" if train_cfg.data_cutoff_idx is None else f"{train_cfg.data_cutoff_idx}samples"
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    epochs = f"ep{train_cfg.epochs}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    optimizer_name = f"opt{train_cfg.optimizer_type}"
    activation = f"act{train_cfg.activation_function}"
    date = time.strftime("%m%d")

    return f"CAT_{num_gpus}_{dataset_size}_{batch_size}_{epochs}_{learning_rate}_{optimizer_name}_{activation}_{date}"

def get_generator():
    g = torch.Generator()
    g.manual_seed(0)  # Fixed global seed
    return g

def get_optimizer(model, train_cfg):
    """Enhanced optimizer selection with different optimizers and scheduling"""
    param_groups = [
        {'params': list(model.MP.parameters()), 'lr': train_cfg.lr_mp, 'weight_decay': train_cfg.weight_decay_mp},
        {'params': list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 
         'lr': train_cfg.lr_backbones, 'weight_decay': train_cfg.weight_decay_backbone}
    ]
    
    if train_cfg.optimizer_type == 'AdamW':
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    elif train_cfg.optimizer_type == 'Adam':
        optimizer = optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)
    elif train_cfg.optimizer_type == 'SGD':
        optimizer = optim.SGD(param_groups, momentum=0.9, nesterov=True)
    elif train_cfg.optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(param_groups, alpha=0.99, eps=1e-8, momentum=0.9)
    elif train_cfg.optimizer_type == 'AdaGrad':
        optimizer = optim.Adagrad(param_groups, eps=1e-10)
    else:
        raise ValueError(f"Unsupported optimizer: {train_cfg.optimizer_type}")
    
    return optimizer

def save_checkpoint(model, optimizer, epoch, global_step, best_accuracy, train_cfg, vlm_cfg, 
                   avg_train_loss=None, checkpoint_dir=None, time_tracker=None):
    """Save comprehensive checkpoint including model, optimizer, and training state"""
    if not is_master():
        return
    
    if checkpoint_dir is None:
        checkpoint_dir = vlm_cfg.vlm_checkpoint_path
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Unwrap model if using DDP
    model_to_save = model.module if is_dist() else model
    
    # Save the model using HuggingFace format
    model_to_save.save_pretrained(checkpoint_dir)
    
    # Save training state
    checkpoint_state = {
        'epoch': epoch,
        'global_step': global_step,
        'best_accuracy': best_accuracy,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_cfg': asdict(train_cfg),
        'vlm_cfg': asdict(vlm_cfg),
        'avg_train_loss': avg_train_loss,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': numpy.random.get_state(),
        'random_rng_state': random.getstate(),
    }
    
    if torch.cuda.is_available():
        checkpoint_state['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    
    # Save time complexity data
    if time_tracker:
        checkpoint_state['time_complexity_data'] = {
            'training_times': time_tracker.training_times,
            'batch_times': time_tracker.batch_times,
            'memory_usage': time_tracker.memory_usage,
            'throughput_data': time_tracker.throughput_data,
            'accuracy_history': time_tracker.accuracy_history
        }
    
    # Save checkpoint state
    checkpoint_path = os.path.join(checkpoint_dir, 'training_state.pt')
    torch.save(checkpoint_state, checkpoint_path)
    
    # Save latest checkpoint info
    latest_info = {
        'checkpoint_dir': checkpoint_dir,
        'epoch': epoch,
        'global_step': global_step,
        'best_accuracy': best_accuracy
    }
    
    latest_path = os.path.join(os.path.dirname(checkpoint_dir), 'latest_checkpoint.json')
    with open(latest_path, 'w') as f:
        json.dump(latest_info, f, indent=2)
    
    print(f"Checkpoint saved at epoch {epoch}, step {global_step} to {checkpoint_dir}")

def load_checkpoint(checkpoint_dir, model, optimizer, device):
    """Load checkpoint and return training state"""
    checkpoint_path = os.path.join(checkpoint_dir, 'training_state.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint from {checkpoint_dir}")
    
    # Load the model using HuggingFace format
    model_to_load = model.module if is_dist() else model
    try:
        loaded_model = CAT.from_pretrained(checkpoint_dir)
        model_to_load.load_state_dict(loaded_model.state_dict())
        print("Model loaded successfully from checkpoint")
    except Exception as e:
        print(f"Warning: Could not load model from checkpoint: {e}")
    
    # Load training state
    checkpoint_state = torch.load(checkpoint_path, map_location=device)
    
    # Restore optimizer state
    try:
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        print("Optimizer state loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load optimizer state: {e}")
    
    # Restore RNG states
    try:
        torch.set_rng_state(checkpoint_state['torch_rng_state'])
        numpy.random.set_state(checkpoint_state['numpy_rng_state'])
        random.setstate(checkpoint_state['random_rng_state'])
        if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint_state:
            torch.cuda.set_rng_state_all(checkpoint_state['cuda_rng_state'])
        print("RNG states restored")
    except Exception as e:
        print(f"Warning: Could not restore RNG states: {e}")
    
    return {
        'epoch': checkpoint_state['epoch'],
        'global_step': checkpoint_state['global_step'],
        'best_accuracy': checkpoint_state['best_accuracy'],
        'avg_train_loss': checkpoint_state.get('avg_train_loss', None),
        'time_complexity_data': checkpoint_state.get('time_complexity_data', {})
    }

def find_latest_checkpoint(base_dir):
    """Find the latest checkpoint directory"""
    latest_path = os.path.join(base_dir, 'latest_checkpoint.json')
    
    if os.path.exists(latest_path):
        with open(latest_path, 'r') as f:
            latest_info = json.load(f)
        return latest_info['checkpoint_dir']
    
    return None

def get_dataloaders(train_cfg, vlm_cfg):
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    # Load and combine all training datasets
    combined_train_data = []
    
    # Fix: Load the main dataset first, then select specific subsets
    try:
        # Load the main cauldron dataset
        cauldron_dataset = load_dataset(train_cfg.train_dataset_path)
        
        # Select and combine specified dataset names
        for dataset_name in train_cfg.train_dataset_name:
            if dataset_name in cauldron_dataset:
                print(f"Loading subset: {dataset_name}")
                combined_train_data.append(cauldron_dataset[dataset_name])
            else:
                print(f"Warning: Dataset '{dataset_name}' not found in {train_cfg.train_dataset_path}")
                # Try alternative approach - load as separate dataset
                try:
                    subset_ds = load_dataset(train_cfg.train_dataset_path, dataset_name, split='train')
                    combined_train_data.append(subset_ds)
                    print(f"Successfully loaded {dataset_name} as separate dataset")
                except Exception as e:
                    print(f"Failed to load {dataset_name}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error loading main dataset: {e}")
        # Fallback: try loading each dataset individually
        for dataset_name in train_cfg.train_dataset_name:
            try:
                # Try different loading approaches
                subset_ds = load_dataset(train_cfg.train_dataset_path, name=dataset_name, split='train')
                combined_train_data.append(subset_ds)
                print(f"Successfully loaded {dataset_name}")
            except Exception as e2:
                try:
                    # Alternative approach
                    subset_ds = load_dataset(f"{train_cfg.train_dataset_path}/{dataset_name}", split='train')
                    combined_train_data.append(subset_ds)
                    print(f"Successfully loaded {dataset_name} with alternative path")
                except Exception as e3:
                    print(f"Failed to load {dataset_name}: {e3}")
                    continue
    
    if not combined_train_data:
        raise ValueError("No training datasets could be loaded successfully")
    
    # Combine all successfully loaded datasets
    if len(combined_train_data) == 1:
        train_ds = combined_train_data[0]
    else:
        train_ds = concatenate_datasets(combined_train_data)
    
    print(f"Combined training dataset size: {len(train_ds)}")
    
    # Load test dataset
    try:
        test_ds = load_dataset(train_cfg.test_dataset_path)
    except Exception as e:
        print(f"Error loading test dataset {train_cfg.test_dataset_path}: {e}")
        # Try alternative loading
        test_ds = load_dataset(train_cfg.test_dataset_path, split='validation')
    
    # Shuffle the training dataset
    train_ds = train_ds.shuffle(seed=0)

    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # Use the entire dataset
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    # Create dataset splits
    train_dataset = VQADataset(train_ds.select(range(train_size)), tokenizer, image_processor)
    val_dataset = VQADataset(train_ds.select(range(train_size, total_samples)), tokenizer, image_processor)
    
    # Handle test dataset structure
    if isinstance(test_ds, dict):
        if 'val' in test_ds:
            test_dataset = MMStarDataset(test_ds['val'], tokenizer, image_processor)
        elif 'test' in test_ds:
            test_dataset = MMStarDataset(test_ds['test'], tokenizer, image_processor)
        elif 'validation' in test_ds:
            test_dataset = MMStarDataset(test_ds['validation'], tokenizer, image_processor)
        else:
            # Take the first available split
            first_split = list(test_ds.keys())[0]
            test_dataset = MMStarDataset(test_ds[first_split], tokenizer, image_processor)
            print(f"Using '{first_split}' split for testing")
    else:
        test_dataset = MMStarDataset(test_ds, tokenizer, image_processor)

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    mmstar_collator = MMStarCollator(tokenizer)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = get_generator()

    # Create dataloaders with optimized settings
    train_sampler = DistributedSampler(
        train_dataset, 
        rank=get_rank(),
        num_replicas=get_world_size(),
    ) if is_dist() else RandomSampler(train_dataset, generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,    # =per device BS in DDP
        sampler=train_sampler,
        collate_fn=vqa_collator,
        num_workers=getattr(train_cfg, 'num_workers', 4),  # Default to 4 if not specified
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True if getattr(train_cfg, 'num_workers', 4) > 0 else False,
        prefetch_factor=2 if getattr(train_cfg, 'num_workers', 4) > 0 else 2,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False  # Usually False for validation
    ) if is_dist() else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        sampler=val_sampler,
        shuffle=(val_sampler is None),  # Only shuffle if not using distributed sampler
        collate_fn=vqa_collator,
        num_workers=getattr(train_cfg, 'num_workers', 4),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True if getattr(train_cfg, 'num_workers', 4) > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=train_cfg.mmstar_batch_size, 
        shuffle=False, 
        collate_fn=mmstar_collator,
        pin_memory=True,
        num_workers=getattr(train_cfg, 'num_workers', 4),
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader, test_loader

def test_mmstar(model, tokenizer, test_loader, device, time_tracker):
    """Enhanced testing with time complexity tracking"""
    time_tracker.start_timer('testing')
    
    total_examples = 0
    correct_predictions = 0
    batch_processing_times = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_start_time = time.time()
            
            image = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            correct_answer = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Forward pass timing
            forward_start = time.time()
            gen = model.generate(input_ids, image, attention_mask)
            forward_time = time.time() - forward_start
            
            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)
            
            is_correct = utils.check_multiple_choice_with_regex(model_output, correct_answer)
            
            total_examples += len(is_correct)
            if is_correct:
                correct_predictions += sum(is_correct)
            
            batch_time = time.time() - batch_start_time
            batch_processing_times.append(batch_time)
            
            # Log memory usage periodically
            if batch_idx % 10 == 0:
                time_tracker.log_memory_usage()
    
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    
    testing_time = time_tracker.stop_timer('testing')
    time_tracker.accuracy_history.append(accuracy)
    
    # Calculate testing throughput
    avg_batch_time = np.mean(batch_processing_times)
    samples_per_second = test_loader.batch_size / avg_batch_time if avg_batch_time > 0 else 0
    
    print(f"Testing completed: {total_examples} samples, {testing_time:.2f}s total, {samples_per_second:.2f} samples/s")
    
    return accuracy

# Enhanced cosine learning rate schedule with warmup
def get_lr(it, max_lr, max_steps, warmup_ratio=0.05, min_lr_ratio=0.1):
    """Enhanced learning rate scheduler with configurable parameters"""
    min_lr = max_lr * min_lr_ratio
    warmup_steps = max_steps * warmup_ratio
    
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def calculate_model_complexity(model):
    """Calculate model complexity metrics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate FLOPs estimation (simplified)
    def count_conv_flops(module):
        if hasattr(module, 'weight') and len(module.weight.shape) == 4:
            # Conv2d: output_h * output_w * kernel_h * kernel_w * in_channels * out_channels
            return module.weight.numel() * 2  # Simplified estimation
        return 0
    
    def count_linear_flops(module):
        if hasattr(module, 'weight') and len(module.weight.shape) == 2:
            return module.weight.numel() * 2
        return 0
    
    total_flops = 0
    for module in model.modules():
        total_flops += count_conv_flops(module)
        total_flops += count_linear_flops(module)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'estimated_flops': total_flops,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

def train(train_cfg, vlm_cfg):
    # Initialize time complexity tracker
    time_tracker = TimeComplexityTracker()
    
    train_loader, val_loader, test_loader = get_dataloaders(train_cfg, vlm_cfg)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    total_dataset_size = len(train_loader.dataset)
    
    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        model = CAT.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = CAT(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    
    # Calculate model complexity
    model_complexity = calculate_model_complexity(model)
    
    # Enhanced optimizer initialization
    optimizer = get_optimizer(model, train_cfg)
    all_params = [p for group in optimizer.param_groups for p in group['params']]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize training state
    start_epoch = 0
    global_step = 0
    best_accuracy = 0
    resume_info = None
    
    # Check for existing checkpoint to resume from
    if train_cfg.auto_resume:
        latest_checkpoint = find_latest_checkpoint(os.path.dirname(vlm_cfg.vlm_checkpoint_path))
        if latest_checkpoint:
            resume_info = load_checkpoint(latest_checkpoint, model, optimizer, device)
            if resume_info:
                start_epoch = resume_info['epoch']
                global_step = resume_info['global_step']
                best_accuracy = resume_info['best_accuracy']
                
                # Restore time complexity data if available
                if 'time_complexity_data' in resume_info:
                    time_data = resume_info['time_complexity_data']
                    time_tracker.training_times = time_data.get('training_times', [])
                    time_tracker.batch_times = time_data.get('batch_times', [])
                    time_tracker.memory_usage = time_data.get('memory_usage', [])
                    time_tracker.throughput_data = time_data.get('throughput_data', [])
                    time_tracker.accuracy_history = time_data.get('accuracy_history', [])
                
                print(f"Resumed training from epoch {start_epoch}, step {global_step}, best accuracy: {best_accuracy:.4f}")
    
    if train_cfg.compile:
        model = torch.compile(model)
    if is_dist():
        model = wrap_model(model)

    # Initialize wandb with enhanced logging
    if train_cfg.log_wandb and is_master():
        run_name = get_run_name(train_cfg)
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
        
        # Add resume info to run name if resuming
        if resume_info:
            run_name += f"_resumed_ep{start_epoch}"
            
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="CAT",
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg),
                "ModelComplexity": model_complexity
            },
            name=run_name,
            resume="allow" if resume_info else None
        )

    if is_master():
        print(f"=== Model Complexity Analysis ===")
        print(f"Total parameters: {model_complexity['total_params']:,}")
        print(f"Trainable parameters: {model_complexity['trainable_params']:,}")
        print(f"Estimated FLOPs: {model_complexity['estimated_flops']:,}")
        print(f"Model size: {model_complexity['model_size_mb']:.2f} MB")
        print(f"Optimizer: {train_cfg.optimizer_type}")
        print(f"Activation function: {train_cfg.activation_function}")
        print(f"Input layer size: {vlm_cfg.vit_img_size}x{vlm_cfg.vit_img_size}")
        print(f"Max sequence length: {vlm_cfg.lm_max_length}")
        print(f"=== Training Configuration ===")
        print(f"Batch size (per device): {train_cfg.batch_size}")
        print(f"Effective batch size: {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}")
        print(f"Learning rates: MP={train_cfg.lr_mp}, Backbones={train_cfg.lr_backbones}")
        print(f"Weight decay: MP={train_cfg.weight_decay_mp}, Backbone={train_cfg.weight_decay_backbone}")
        print(f"Epochs: {train_cfg.epochs}")
        print(f"Number of workers: {train_cfg.num_workers}")
        print(f"CAT initialized with {sum(p.numel() for p in model.parameters()):,} parameters") 
        print(f"Training summary{' (global)' if is_dist() else ''}: {len(train_loader.dataset)} samples, {int(len(train_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Training summary per GPU: {len(train_loader)} batches/epoch, batch size {train_loader.batch_size}")
        print(f"Validation summary{' (global)' if is_dist() else ''}: {len(val_loader.dataset)} samples, {int(len(val_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Validation summary per GPU: {len(val_loader)} batches/epoch, batch size {val_loader.batch_size}")

    epoch_times = []
    
    for epoch in range(start_epoch, train_cfg.epochs):
        time_tracker.start_timer('training')
        epoch_start_time = time.time()

        image = batch['images'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
        # Forward pass with timing
        time_tracker.start_timer('forward')
        loss = model(input_ids, image, attention_mask, labels)
        loss = loss / train_cfg.gradient_accumulation_steps
        forward_time = time_tracker.stop_timer('forward')
            
         # Backward pass with timing
        time_tracker.start_timer('backward')
        loss.backward()
        backward_time = time_tracker.stop_timer('backward')
            
        total_train_loss += loss.item() * train_cfg.gradient_accumulation_steps
            
        # Count tokens for throughput calculation
        batch_tokens = (labels != -100).sum().item()
        total_tokens_processed += batch_tokens
            
        # Gradient accumulation and optimization step
        if (i + 1) % train_cfg.gradient_accumulation_steps == 0:
            time_tracker.start_timer('optimizer')
                
            # Gradient clipping if enabled
            if train_cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(all_params, train_cfg.grad_clip_norm)
                
            # Update learning rates with scheduler
            current_lr_mp = get_lr(global_step, train_cfg.lr_mp, 
                                    train_cfg.epochs * len(train_loader) // train_cfg.gradient_accumulation_steps)
            current_lr_backbone = get_lr(global_step, train_cfg.lr_backbones,
                                        train_cfg.epochs * len(train_loader) // train_cfg.gradient_accumulation_steps)
                
            for param_group in optimizer.param_groups:
                if param_group == optimizer.param_groups[0]:  # MP parameters
                    param_group['lr'] = current_lr_mp
                else:  # Backbone parameters
                    param_group['lr'] = current_lr_backbone
                
            optimizer.step()
            optimizer.zero_grad()
            optimizer_time = time_tracker.stop_timer('optimizer')
                
            global_step += 1
                
            # Log memory usage
            memory_usage = time_tracker.log_memory_usage()
                
            # Calculate throughput
            batch_time = time.time() - batch_start_time
            time_tracker.batch_times.append(batch_time)
            samples_per_second = train_cfg.batch_size * get_world_size() / batch_time
            tokens_per_second = batch_tokens / forward_time if forward_time > 0 else 0
            time_tracker.log_throughput(train_cfg.batch_size * get_world_size(), 
                                          tokens_per_second, samples_per_second)
                
            # Detailed logging
            if global_step % train_cfg.log_every == 0 and is_master():
                avg_loss = total_train_loss / (i + 1)
                print(f"Epoch {epoch+1}/{train_cfg.epochs}, Step {global_step}, "
                          f"Loss: {avg_loss:.6f}, LR_MP: {current_lr_mp:.2e}, "
                          f"LR_Backbone: {current_lr_backbone:.2e}, "
                          f"Memory: {memory_usage:.1f}MB, "
                          f"Samples/s: {samples_per_second:.2f}, "
                          f"Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s")
                    
                if train_cfg.log_wandb:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/lr_mp': current_lr_mp,
                            'train/lr_backbone': current_lr_backbone,
                            'train/memory_usage_mb': memory_usage,
                            'train/samples_per_second': samples_per_second,
                            'train/tokens_per_second': tokens_per_second,
                            'train/forward_time': forward_time,
                            'train/backward_time': backward_time,
                            'train/optimizer_time': optimizer_time,
                            'train/batch_time': batch_time,
                            'train/epoch': epoch,
                            'train/global_step': global_step
                        })
        
        # End of epoch processing
        avg_train_loss = total_train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        time_tracker.epoch_times.append(epoch_time)
        time_tracker.loss_history.append(avg_train_loss)
        training_time = time_tracker.stop_timer('training')
        
        if is_master():
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_train_loss:.6f}")
        
        # Validation phase
        if (epoch + 1) % train_cfg.val_every == 0:
            model.eval()
            val_start_time = time.time()
            total_val_loss = 0
            val_samples = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    image = val_batch['images'].to(device, non_blocking=True)
                    input_ids = val_batch['input_ids'].to(device, non_blocking=True)
                    labels = val_batch['labels'].to(device, non_blocking=True)
                    attention_mask = val_batch['attention_mask'].to(device, non_blocking=True)
                    
                    val_loss = model(input_ids, image, attention_mask, labels)
                    total_val_loss += val_loss.item()
                    val_samples += input_ids.size(0)
            
            # Gather validation results from all processes
            if is_dist():
                val_loss_tensor = torch.tensor(total_val_loss, device=device)
                val_samples_tensor = torch.tensor(val_samples, device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_samples_tensor, op=dist.ReduceOp.SUM)
                total_val_loss = val_loss_tensor.item()
                val_samples = val_samples_tensor.item()
            
            avg_val_loss = total_val_loss / len(val_loader) if is_dist() else total_val_loss / len(val_loader)
            val_time = time.time() - val_start_time
            
            if is_master():
                print(f"Validation Loss: {avg_val_loss:.6f}, Time: {val_time:.2f}s")
                
                if train_cfg.log_wandb:
                    wandb.log({
                        'val/loss': avg_val_loss,
                        'val/time': val_time,
                        'train/epoch': epoch
                    })
        
        # Testing phase
        if (epoch + 1) % train_cfg.test_every == 0:
            model.eval()
            test_accuracy = test_mmstar(model, tokenizer, test_loader, device, time_tracker)
            
            # Gather test results from all processes
            if is_dist():
                accuracy_tensor = torch.tensor(test_accuracy, device=device)
                dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)
                test_accuracy = accuracy_tensor.item() / get_world_size()
            
            if is_master():
                print(f"Test Accuracy: {test_accuracy:.4f}")
                
                if train_cfg.log_wandb:
                    wandb.log({
                        'test/accuracy': test_accuracy,
                        'train/epoch': epoch
                    })
                
                # Save checkpoint if this is the best model
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    checkpoint_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, f"best_model_epoch_{epoch+1}")
                    save_checkpoint(model, optimizer, epoch + 1, global_step, best_accuracy, 
                                  train_cfg, vlm_cfg, avg_train_loss, checkpoint_dir, time_tracker)
                    print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        # Regular checkpoint saving
        if (epoch + 1) % train_cfg.save_every == 0 and is_master():
            checkpoint_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, f"epoch_{epoch+1}")
            save_checkpoint(model, optimizer, epoch + 1, global_step, best_accuracy,
                          train_cfg, vlm_cfg, avg_train_loss, checkpoint_dir, time_tracker)
    
    # Final training completion
    if is_master():
        print("Training completed!")
        
        # Generate comprehensive performance analysis
        complexity_analysis = time_tracker.analyze_complexity()
        print("\n=== Training Performance Analysis ===")
        print(f"Total training time: {complexity_analysis['total_training_time']:.2f}s")
        print(f"Average epoch time: {np.mean(epoch_times):.2f}s")
        print(f"Average batch processing time: {complexity_analysis['avg_batch_time']:.4f}s")
        print(f"Average forward pass time: {complexity_analysis['avg_forward_time']:.4f}s")
        print(f"Average backward pass time: {complexity_analysis['avg_backward_time']:.4f}s")
        print(f"Peak GPU memory usage: {complexity_analysis['peak_memory_mb']:.1f}MB")
        print(f"Training efficiency: {complexity_analysis['training_efficiency']:.2f} batches/second")
        print(f"Best test accuracy achieved: {best_accuracy:.4f}")
        
        # Generate performance plots
        time_tracker.plot_performance_metrics("training_performance_analysis.png")
        
        # Log final results to wandb
        if train_cfg.log_wandb:
            wandb.log({
                'final/best_accuracy': best_accuracy,
                'final/total_training_time': complexity_analysis['total_training_time'],
                'final/avg_epoch_time': np.mean(epoch_times),
                'final/peak_memory_mb': complexity_analysis['peak_memory_mb'],
                'final/training_efficiency': complexity_analysis['training_efficiency']
            })
            
            # Upload performance plots
            wandb.log({"performance_analysis": wandb.Image("training_performance_analysis.png")})
            wandb.finish()
        
        # Save final checkpoint
        final_checkpoint_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, "final_model")
        save_checkpoint(model, optimizer, train_cfg.epochs, global_step, best_accuracy,
                      train_cfg, vlm_cfg, avg_train_loss, final_checkpoint_dir, time_tracker)
        
        # Save time complexity analysis
        with open(os.path.join(vlm_cfg.vlm_checkpoint_path, "training_analysis.json"), 'w') as f:
            json.dump({
                'model_complexity': model_complexity,
                'performance_analysis': complexity_analysis,
                'training_config': asdict(train_cfg),
                'vlm_config': asdict(vlm_cfg)
            }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Train Vision-Language Model with Time Complexity Analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to training config file')
    parser.add_argument('--vlm_config', type=str, required=True, help='Path to VLM config file')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        train_cfg_dict = json.load(f)
    train_cfg = config.TrainConfig(**train_cfg_dict)
    
    with open(args.vlm_config, 'r') as f:
        vlm_cfg_dict = json.load(f)
    vlm_cfg = config.VLMConfig(**vlm_cfg_dict)
    
    # Initialize distributed training if requested
    if args.distributed:
        init_dist()
    
    try:
        train(train_cfg, vlm_cfg)
    finally:
        if args.distributed:
            destroy_dist()

if __name__ == "__main__":
    main()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()

        # Set epoch for distributed sampler
        if is_dist():
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            time_tracker.start_timer('data_loading')
            batch_start_time = time.time()
