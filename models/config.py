from dataclasses import dataclass


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 224
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = 'google/siglip-base-patch16-224'

    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_vocab_size: int = 49152
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    IMAGE_TOKEN_LENGTH: int = 49
    TOTAL_SEQUENCE_LENGTH: int = 128
    lm_max_length: int = TOTAL_SEQUENCE_LENGTH - IMAGE_TOKEN_LENGTH  
    lm_use_tokens: bool = False 
    lm_tie_weights: bool = True 
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M'
    lm_tokenizer: str = 'HuggingFaceTB/cosmo2-tokenizer'
    lm_eos_token_id: int = 0

    mp_pixel_shuffle_factor: int = 2

    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'checkpoints/CAT-222M'
    hf_repo_name: str = 'CAT'


@dataclass
class TrainConfig:
    lr_mp: float = 2e-3
    lr_backbones: float = 1e-4
    data_cutoff_idx: int = None
    val_ratio: float = 0.025
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    mmstar_batch_size: int = 32
    max_grad_norm: float = None
    eval_in_epochs: bool = True
    eval_interval: int = 500
    epochs: int = 10
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    train_dataset_name: tuple[str, ...] = ("chart2text", "chartqa", "figureqa")
    test_dataset_path: str = "Lin-Chen/MMStar"
    wandb_entity: str = "mwz" # Indicate the entity to log to in wandb
    log_wandb: bool = True
