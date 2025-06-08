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
    date = time.strftime("%m%d")

    return f"CAT_{num_gpus}_{dataset_size}_{batch_size}_{epochs}_{learning_rate}_{date}"

def get_generator():
    g = torch.Generator()
    g.manual_seed(0)  # Fixed global seed
    return g

def save_checkpoint(model, optimizer, epoch, global_step, best_accuracy, train_cfg, vlm_cfg, 
                   avg_train_loss=None, checkpoint_dir=None):
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
        'avg_train_loss': checkpoint_state.get('avg_train_loss', None)
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
    for dataset_name in train_cfg.train_dataset_name:
        train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)
        combined_train_data.append(train_ds['train'])
    train_ds = concatenate_datasets(combined_train_data)
    
    test_ds = load_dataset(train_cfg.test_dataset_path)
    train_ds = train_ds.shuffle(seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatinated datasets

    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # Use the entire dataset
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    train_dataset = VQADataset(train_ds.select(range(train_size)), tokenizer, image_processor)
    val_dataset = VQADataset(train_ds.select(range(train_size, total_samples)), tokenizer, image_processor)
    test_dataset = MMStarDataset(test_ds['val'], tokenizer, image_processor)

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    mmstar_collator = MMStarCollator(tokenizer)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = get_generator()

    # Create dataloaders
    train_sampler = DistributedSampler(
        train_dataset, 
        rank=get_rank(),
        num_replicas=get_world_size(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,    # =per device BS in DDP
        sampler=train_sampler,
        collate_fn=vqa_collator,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False  # Usually False for validation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        sampler=val_sampler,
        collate_fn=vqa_collator,
        num_workers=0,
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
        worker_init_fn=seed_worker,
        generator=g,
        )

    return train_loader, val_loader, test_loader

def test_mmstar(model, tokenizer, test_loader, device):
    total_examples = 0
    correct_predictions = 0
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

# Cosine learning rate schedule with warmup (from Karpathy)
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
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

def train(train_cfg, vlm_cfg):
    train_loader, val_loader, test_loader = get_dataloaders(train_cfg, vlm_cfg)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    total_dataset_size = len(train_loader.dataset)
    
    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        model = CAT.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = CAT(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    
    # Define optimizer groups
    param_groups = [{'params': list(model.MP.parameters()), 'lr': train_cfg.lr_mp},
                    {'params': list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 'lr': train_cfg.lr_backbones}]
    optimizer = optim.AdamW(param_groups)
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
                print(f"Resumed training from epoch {start_epoch}, step {global_step}, best accuracy: {best_accuracy:.4f}")
    
    if train_cfg.compile:
        model = torch.compile(model)
    if is_dist():
        model = wrap_model(model)

    # Initialize wandb
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
                "TrainConfig": asdict(train_cfg)
            },
            name=run_name,
            resume="allow" if resume_info else None
        )

    if is_master():
        print(f"CAT initialized with {sum(p.numel() for p in model.parameters()):,} parameters") 
        print(f"Training summary{' (global)' if is_dist() else ''}: {len(train_loader.dataset)} samples, {int(len(train_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Training summary per GPU: {len(train_loader)} batches/epoch, batch size {train_loader.batch_size}")
        print(f"Validation summary{' (global)' if is_dist() else ''}: {len(val_loader.dataset)} samples, {int(len(val_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Validation summary per GPU: {len(val_loader)} batches/epoch, batch size {val_loader.batch_size}")

    epoch_times = []
    
    for epoch in range(start_epoch, train_cfg.epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
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

            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
            if (is_dist()
                and train_cfg.gradient_accumulation_steps > 1
                and not (
                    (i + 1) % train_cfg.gradient_accumulation_steps == 0 
                    or i + 1 == len(train_loader)
                )):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # Set to float16 if your hardware doesn't support bfloat16
                with context:
                    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()

            if (i + 1) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader):
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)

                adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, len(train_loader) * train_cfg.epochs)
                adj_lr_backbones = get_lr(global_step, train_cfg.lr_backbones, len(train_loader) * train_cfg.epochs)
                optimizer.param_groups[0]['lr'] = adj_lr_mp
                optimizer.param_groups[1]['lr'] = adj_lr_backbones
                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item() # Sum of attention mask gives number of tokens
            num_tokens += images.shape[0] * ((images.shape[2] / vlm_cfg.vit_patch_size) ** 2) / (vlm_cfg.mp_pixel_shuffle_factor ** 2) # Add image tokens = batch_size * (((img_size / patch_size) ** 2) / (pixel_shuffle_factor ** 2))
            total_tokens_processed += num_tokens

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = num_tokens / batch_duration 

            # gather loss and t/s from all ranks if DDP
            batch_loss = mean(dist_gather(batch_loss)) if is_dist() else batch_loss  
            tokens_per_second = sum(dist_gather(tokens_per_second)) if is_dist() else tokens_per_second  

            if train_cfg.eval_in_epochs and global_step % train_cfg.eval_interval == 0: #and is_master():
                model.eval()
                torch.cuda.empty_cache()  # Clear GPU memory
                with torch.no_grad():
                    total_val_loss = 0
                    for batch in val_loader:
                        images = batch["image"].to(device)
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

                        total_val_loss += loss.item()
                    avg_val_loss = total_val_loss / len(val_loader)
                    avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss
                    if train_cfg.log_wandb and is_master():
                        run.log({"val_loss": avg_val_loss}, step=global_step)

                    if is_master() and global_step % (train_cfg.eval_interval*2) == 0:
                        eval_model = model.module if is_dist() else model  # unwrap the model for eval if DDP
                        epoch_accuracy = test_mmstar(eval_model, tokenizer, test_loader, device)
                        if epoch_accuracy > best_accuracy:
                            best_accuracy = epoch_accuracy
                            # Save best model checkpoint
                            checkpoint_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, f"best_checkpoint_step_{global_step}")
                            save_checkpoint(eval_model, optimizer, epoch, global_step, best_accuracy, 
                                          train_cfg, vlm_cfg, checkpoint_dir=checkpoint_dir)
                            
                            # Also save to the main checkpoint path for backward compatibility
                            eval_model.save_pretrained(save_directory=vlm_cfg.vlm_checkpoint_path)
                            
                        if train_cfg.log_wandb and is_master():    
                            run.log({"accuracy": epoch_accuracy}, step=global_step)
                        print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}, Accuracy: {epoch_accuracy:.4f}")
                    elif is_master() and not global_step % (train_cfg.eval_interval*4) == 0:
                        print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}")

                model.train()          

            if train_cfg.log_wandb and is_master():
                run.log({
                    "batch_loss": batch_loss,
                    "tokens_per_second": tokens_per_second,
                    **({"grad_norm": grad_norm} if train_cfg.max_grad_norm is not None else {})
                }, step=global_step)
                
            if (i + 1) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader):
                global_step += 1

            # Save checkpoint periodically
            if (train_cfg.save_checkpoint_steps > 0 and 
                global_step % train_cfg.save_checkpoint_steps == 0 and
                is_master()):
                checkpoint_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, f"checkpoint_step_{global_step}")
                save_checkpoint(model, optimizer, epoch, global_step, best_accuracy, 
                              train_cfg, vlm_cfg, total_train_loss / max(1, i + 1), checkpoint_dir)

        avg_train_loss = total_train_loss / len(train_loader)
        # gather average batch loss from all ranks if DDP
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss  

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed accross all ranks if DDP
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed  
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_cfg.log_wandb:
                run.log({"epoch_loss": avg_train_loss,
                         "epoch_duration": epoch_duration,
                         "epoch_tokens_per_second": epoch_tokens_per_second,
                         "epoch": epoch})

            print(f"Epoch {epoch+1}/{train_cfg.epochs}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")

        # Save checkpoint at the end of each epoch
        if is_master():
            checkpoint_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, f"checkpoint_epoch_{epoch+1}")
            save_checkpoint(model, optimizer, epoch + 1, global_step, best_accuracy, 
                          train_cfg, vlm_cfg, avg_train_loss, checkpoint_dir)

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        total_samples_processed = len(train_loader.dataset) * train_cfg.epochs
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # Save final checkpoint
        final_checkpoint_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, "final_checkpoint")
        save_checkpoint(model, optimizer, train_cfg.epochs, global_step, best_accuracy, 
                       train_cfg, vlm_cfg, checkpoint_dir=final_checkpoint_dir)

        # Push the best model to the hub (Please set your user name in the config!)
        if vlm_cfg.hf_repo_name is not None:
            print("Training complete. Pushing model to Hugging Face Hub...")
            try:
                # Load the best checkpoint for pushing to hub
                best_checkpoint_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, "best_checkpoint")
                if os.path.exists(best_checkpoint_dir):
                    hf_model = CAT.from_pretrained(best_checkpoint_dir)
                else:
                    hf_model = CAT.from_pretrained(vlm_cfg.vlm_checkpoint_path)
                
                hf_model.push_to_hub(vlm_cfg.hf_repo_name)
                print(f"Model successfully pushed to {vlm_cfg.hf_repo_name}")
            except Exception as e:
                print(f"Error pushing model to HuggingFace Hub: {e}")

        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.summary["mmstar_acc"] = best_accuracy
            run.summary["final_epoch"] = train_cfg.epochs
            run.summary["total_steps"] = global_step
            run.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_backbones', type=float, help='Learning rate for the backbones')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint for loading or saving')
    parser.add_argument('--compile', type=bool, help='Use torch.compile to optimize the model')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')
    parser.add_argument('--auto_resume', type=bool, default=True, help='Automatically resume from the latest checkpoint if available')
    parser.add_argument('--save_checkpoint_steps', type=int, default=1000, help='Save checkpoint every N steps (0 to disable)')

    args = parser.parse_args()

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()

    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_backbones is not None:
        train_cfg.lr_backbones = args.lr_backbones
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.auto_resume is not None:
        train_cfg.auto_resume = args.auto_resume
    if args.save_checkpoint_steps is not None:
        train_cfg.save_checkpoint_steps = args.save_checkpoint_steps

    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)

    train(train_cfg, vlm_cfg)

    if is_dist():
        destroy_dist()

if __name__ == "__main__":
    main() 
