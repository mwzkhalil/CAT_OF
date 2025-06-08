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
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.collators import VQACollator, MMStarCollator
from data.datasets import MMStarDataset, VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import CAT
import models.config as config
import models.utils as utils

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Distributed training functions
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

# Training utilities
def get_run_name(train_cfg):
    dataset_size = "full_ds" if train_cfg.data_cutoff_idx is None else f"{train_cfg.data_cutoff_idx}samples"
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    epochs = f"ep{train_cfg.epochs}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d")
    return f"CAT_{num_gpus}_{dataset_size}_{batch_size}_{epochs}_{learning_rate}_{date}"

def get_generator():
    g = torch.Generator()
    g.manual_seed(0)
    return g

def save_checkpoint(model, optimizer, epoch, global_step, best_accuracy, train_cfg, vlm_cfg, 
                   avg_train_loss=None, checkpoint_dir=None, no_improve_count=0):
    if not is_master():
        return
    
    checkpoint_dir = checkpoint_dir or vlm_cfg.vlm_checkpoint_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model
    model_to_save = model.module if is_dist() else model
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
        'no_improve_count': no_improve_count,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': numpy.random.get_state(),
        'random_rng_state': random.getstate(),
    }
    
    if torch.cuda.is_available():
        checkpoint_state['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    
    torch.save(checkpoint_state, os.path.join(checkpoint_dir, 'training_state.pt'))
    
    # Save latest checkpoint info
    latest_info = {
        'checkpoint_dir': checkpoint_dir,
        'epoch': epoch,
        'global_step': global_step,
        'best_accuracy': best_accuracy
    }
    
    with open(os.path.join(os.path.dirname(checkpoint_dir), 'latest_checkpoint.json'), 'w') as f:
        json.dump(latest_info, f, indent=2)
    
    print(f"Checkpoint saved at epoch {epoch}, step {global_step} to {checkpoint_dir}")

def load_checkpoint(checkpoint_dir, model, optimizer, device):
    checkpoint_path = os.path.join(checkpoint_dir, 'training_state.pt')
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint from {checkpoint_dir}")
    checkpoint_state = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    model_to_load = model.module if is_dist() else model
    try:
        loaded_model = CAT.from_pretrained(checkpoint_dir)
        model_to_load.load_state_dict(loaded_model.state_dict())
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading warning: {e}")
    
    # Load optimizer
    try:
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        print("Optimizer state loaded")
    except Exception as e:
        print(f"Optimizer loading warning: {e}")
    
    # Restore RNG states
    try:
        torch.set_rng_state(checkpoint_state['torch_rng_state'])
        numpy.random.set_state(checkpoint_state['numpy_rng_state'])
        random.setstate(checkpoint_state['random_rng_state'])
        if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint_state:
            torch.cuda.set_rng_state_all(checkpoint_state['cuda_rng_state'])
    except Exception as e:
        print(f"RNG restoration warning: {e}")
    
    return {
        'epoch': checkpoint_state['epoch'],
        'global_step': checkpoint_state['global_step'],
        'best_accuracy': checkpoint_state['best_accuracy'],
        'no_improve_count': checkpoint_state.get('no_improve_count', 0),
        'avg_train_loss': checkpoint_state.get('avg_train_loss', None),
        'train_cfg': checkpoint_state.get('train_cfg'),
        'vlm_cfg': checkpoint_state.get('vlm_cfg')
    }

def find_latest_checkpoint(base_dir):
    latest_path = os.path.join(base_dir, 'latest_checkpoint.json')
    if os.path.exists(latest_path):
        with open(latest_path, 'r') as f:
            return json.load(f)['checkpoint_dir']
    return None

# Data handling
def get_dataloaders(train_cfg, vlm_cfg):
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    # Load and combine datasets
    combined_train_data = []
    for dataset_name in train_cfg.train_dataset_name:
        train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)
        combined_train_data.append(train_ds['train'])
    train_ds = concatenate_datasets(combined_train_data)
    test_ds = load_dataset(train_cfg.test_dataset_path)
    train_ds = train_ds.shuffle(seed=0)

    # Dataset sizing
    total_samples = len(train_ds) if train_cfg.data_cutoff_idx is None else min(len(train_ds), train_cfg.data_cutoff_idx)
    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    # Create datasets
    train_dataset = VQADataset(train_ds.select(range(train_size)), tokenizer, image_processor)
    val_dataset = VQADataset(train_ds.select(range(train_size, total_samples)), tokenizer, image_processor)
    test_dataset = MMStarDataset(test_ds['val'], tokenizer, image_processor)

    # Collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    mmstar_collator = MMStarCollator(tokenizer)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = get_generator()

    # Dataloaders
    train_sampler = DistributedSampler(
        train_dataset, 
        rank=get_rank(),
        num_replicas=get_world_size(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        sampler=train_sampler,
        collate_fn=vqa_collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        sampler=val_sampler,
        collate_fn=vqa_collator,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=train_cfg.mmstar_batch_size, 
        shuffle=False, 
        collate_fn=mmstar_collator,
        pin_memory=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader, test_loader

# Evaluation
def test_mmstar(model, tokenizer, test_loader, device):
    total_examples = 0
    correct_predictions = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            image = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            correct_answer = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            gen = model.generate(input_ids, image, attention_mask)
            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)
            
            is_correct = utils.check_multiple_choice_with_regex(model_output, correct_answer)
            
            total_examples += len(is_correct)
            if is_correct:
                correct_predictions += sum(is_correct)
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    return accuracy

# Learning rate scheduler
def get_lr(it, max_lr, max_steps, min_lr_ratio=0.1, warmup_ratio=0.1):
    min_lr = max_lr * min_lr_ratio
    warmup_steps = int(max_steps * warmup_ratio)
    
    # Warmup phase
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    # Decay phase
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)  # Clamp to 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Main training function
def train(train_cfg, vlm_cfg):
    # Initialize distributed training if needed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    # Check for checkpoint to resume
    resume_info = None
    if train_cfg.auto_resume:
        latest_checkpoint = find_latest_checkpoint(os.path.dirname(vlm_cfg.vlm_checkpoint_path))
        if latest_checkpoint:
            print(f"Found checkpoint: {latest_checkpoint}")

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(train_cfg, vlm_cfg)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)
    total_dataset_size = len(train_loader.dataset)
    
    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint and latest_checkpoint:
        model = CAT.from_pretrained(latest_checkpoint)
    else:
        model = CAT(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    
    # Optimizer configuration
    param_groups = [
        {'params': model.MP.parameters(), 'lr': train_cfg.lr_mp},
        {'params': list(model.decoder.parameters()) + 
                   list(model.vision_encoder.parameters()), 
         'lr': train_cfg.lr_backbones, 'weight_decay': 0.01}
    ]
    optimizer = optim.AdamW(
        param_groups, 
        betas=(0.9, 0.98), 
        eps=1e-6,
        weight_decay=0.01
    )
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training state
    start_epoch = 0
    global_step = 0
    best_accuracy = 0
    no_improve_count = 0
    
    # Load checkpoint if available
    if train_cfg.auto_resume and latest_checkpoint:
        resume_info = load_checkpoint(latest_checkpoint, model, optimizer, device)
        if resume_info:
            start_epoch = resume_info['epoch'] + 1
            global_step = resume_info['global_step']
            best_accuracy = resume_info['best_accuracy']
            no_improve_count = resume_info['no_improve_count']
            print(f"Resuming training from epoch {start_epoch}, step {global_step}")
            print(f"Best accuracy: {best_accuracy:.4f}, No improve count: {no_improve_count}")
    
    # Compile and distribute model
    if train_cfg.compile:
        model = torch.compile(model)
    if is_dist():
        model = wrap_model(model)

    # Initialize wandb
    if train_cfg.log_wandb and is_master():
        run_name = get_run_name(train_cfg)
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
        if resume_info:
            run_name += f"_resumed_ep{start_epoch}"
            
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="CAT",
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg)
            },
            name=run_name,
            resume="allow" if resume_info else None
        )

    # Training info
    if is_master():
        print(f"CAT Parameters: {sum(p.numel() for p in model.parameters()):,}") 
        print(f"Training Config: {len(train_loader.dataset)} samples, "
              f"{len(train_loader)} batches/epoch, "
              f"Batch size: {train_loader.batch_size * get_world_size() * train_cfg.gradient_accumulation_steps} "
              f"({get_world_size()} GPUs)")

    # Training loop
    total_steps = len(train_loader) * train_cfg.epochs
    epoch_times = []
    early_stop = False
    
    for epoch in range(start_epoch, train_cfg.epochs):
        if early_stop:
            break
            
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        # Set epoch for distributed sampler
        if is_dist():
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            batch_start_time = time.time()
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Gradient sync context
            if (is_dist() and train_cfg.gradient_accumulation_steps > 1 and
                not ((i + 1) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader))):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            # Forward pass with mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16), context:
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
                loss = loss / train_cfg.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Optimization step
            if (i + 1) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader):
                # Gradient clipping
                if train_cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                
                # Adjust learning rates
                for group_idx, group in enumerate(optimizer.param_groups):
                    if group_idx == 0:  # MP parameters
                        group['lr'] = get_lr(global_step, train_cfg.lr_mp, total_steps)
                    else:  # Backbone parameters
                        group['lr'] = get_lr(global_step, train_cfg.lr_backbones, total_steps)
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Calculate batch metrics
            batch_loss = loss.item() * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss
            batch_duration = time.time() - batch_start_time
            tokens_per_second = (images.size(0) * images.size(1) * images.size(2) / batch_duration

            # Logging
            if is_master() and global_step % train_cfg.log_interval == 0:
                print(f"Epoch: {epoch+1}/{train_cfg.epochs} | "
                      f"Batch: {i+1}/{len(train_loader)} | "
                      f"Loss: {batch_loss:.4f} | "
                      f"Tokens/s: {tokens_per_second:.0f}")

            # Evaluation
            if train_cfg.eval_in_epochs and global_step % train_cfg.eval_interval == 0:
                model.eval()
                torch.cuda.empty_cache()
                
                # Validation loss
                val_loss = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_images = val_batch["image"].to(device)
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device)

                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            _, loss = model(val_input_ids, val_images, 
                                          attention_mask=val_attention_mask, 
                                          targets=val_labels)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                if is_dist():
                    avg_val_loss = mean(dist_gather(avg_val_loss))
                
                # MMStar accuracy
                if is_master() and global_step % (train_cfg.eval_interval * 2) == 0:
                    eval_model = model.module if is_dist() else model
                    accuracy = test_mmstar(eval_model, tokenizer, test_loader, device)
                    
                    # Update best accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        no_improve_count = 0
                        checkpoint_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, f"best_epoch{epoch+1}_acc{accuracy:.4f}")
                        save_checkpoint(model, optimizer, epoch, global_step, 
                                      best_accuracy, train_cfg, vlm_cfg, 
                                      checkpoint_dir=checkpoint_dir,
                                      no_improve_count=no_improve_count)
                    else:
                        no_improve_count += 1
                    
                    # Check early stopping
                    if no_improve_count >= train_cfg.early_stopping_patience:
                        print(f"No improvement for {no_improve_count} evaluations. Early stopping.")
                        early_stop = True
                        break
                    
                    # Log metrics
                    if train_cfg.log_wandb:
                        wandb.log({
                            "val_loss": avg_val_loss,
                            "accuracy": accuracy,
                            "best_accuracy": best_accuracy
                        }, step=global_step)
                    
                    print(f"Validation | Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f} | Best: {best_accuracy:.4f}")
                
                model.train()

        # Epoch statistics
        avg_train_loss = total_train_loss / len(train_loader)
        if is_dist():
            avg_train_loss = mean(dist_gather(avg_train_loss))
        
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        
        if is_master():
            print(f"Epoch {epoch+1} Summary | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Time: {epoch_duration:.2f}s")
            
            # Save epoch checkpoint
            checkpoint_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, f"epoch_{epoch+1}")
            save_checkpoint(model, optimizer, epoch+1, global_step, 
                           best_accuracy, train_cfg, vlm_cfg, 
                           avg_train_loss, checkpoint_dir,
                           no_improve_count)

    # Finalize training
    if is_master():
        # Save final model
        final_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, "final_model")
        save_checkpoint(model, optimizer, train_cfg.epochs, global_step, 
                       best_accuracy, train_cfg, vlm_cfg, 
                       checkpoint_dir=final_dir)
        
        # Performance summary
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        samples_per_second = total_dataset_size / avg_epoch_time
        print(f"\nTraining Complete | Best Accuracy: {best_accuracy:.4f}")
        print(f"Average Epoch Time: {avg_epoch_time:.2f}s | Samples/s: {samples_per_second:.2f}")
        
        # Push to Hugging Face Hub
        if vlm_cfg.hf_repo_name:
            print(f"Pushing model to Hugging Face Hub: {vlm_cfg.hf_repo_name}")
            try:
                model_to_push = model.module if is_dist() else model
                model_to_push.push_to_hub(vlm_cfg.hf_repo_name)
            except Exception as e:
                print(f"Error pushing model: {e}")
        
        if train_cfg.log_wandb:
            wandb.finish()

    if is_dist():
        destroy_dist()

# Main function
def main():
    parser = argparse.ArgumentParser(description="CAT Model Training")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Per-device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr_backbones', type=float, default=1e-5, help='Learning rate for backbones')
    parser.add_argument('--lr_mp', type=float, default=5e-4, help='Learning rate for mapping network')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    
    # Evaluation parameters
    parser.add_argument('--eval_interval', type=int, default=500, help='Steps between evaluations')
    parser.add_argument('--log_interval', type=int, default=100, help='Steps between logging')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience')
    
    # System parameters
    parser.add_argument('--compile', action='store_true', help='Compile model with torch.compile')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Model checkpoint path')
    parser.add_argument('--hf_repo', type=str, help='Hugging Face repository name')
    
    args = parser.parse_args()
    
    # Configuration setup
    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()
    
    # Apply command-line arguments
    train_cfg.batch_size = args.batch_size
    train_cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    train_cfg.epochs = args.epochs
    train_cfg.lr_backbones = args.lr_backbones
    train_cfg.lr_mp = args.lr_mp
    train_cfg.max_grad_norm = args.max_grad_norm
    train_cfg.eval_interval = args.eval_interval
    train_cfg.log_interval = args.log_interval
    train_cfg.early_stopping_patience = args.early_stopping_patience
    train_cfg.compile = args.compile
    train_cfg.auto_resume = args.resume
    vlm_cfg.vlm_checkpoint_path = args.checkpoint_path
    
    if args.hf_repo:
        vlm_cfg.hf_repo_name = args.hf_repo
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_path, exist_ok=True)
    
    # Start training
    train(train_cfg, vlm_cfg)

if __name__ == "__main__":
    main()
