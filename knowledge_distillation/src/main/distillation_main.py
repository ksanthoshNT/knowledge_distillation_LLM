import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import logging
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EnhancedKnowledgeDistillation:
    def __init__(self, args, local_rank):
        self.args = args
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")

        logger.info("Initializing knowledge distillation...")
        self._initialize_models()
        self._setup_model_hooks()

    def _initialize_models(self):
        """Initialize teacher and student models with proper configuration"""
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.teacher_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Initialize teacher model
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.args.teacher_model_name,
                device_map={"": self.device},
                torch_dtype=torch.bfloat16,
                output_hidden_states=True,
                output_attentions=True
            )
            self.teacher_model.eval()  # Always in eval mode

            # Initialize student model
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.args.student_model_name,
                device_map={"": self.device},
                torch_dtype=torch.bfloat16,
                output_hidden_states=True,
                output_attentions=True
            )

            # Wrap student model in DDP
            self.student_model = DDP(
                self.student_model,
                device_ids=[self.local_rank],
                find_unused_parameters=True
            )

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def _setup_model_hooks(self):
        """Set up feature extraction hooks for both models"""
        self.teacher_features: Dict[str, torch.Tensor] = {}
        self.student_features: Dict[str, torch.Tensor] = {}

        def create_hook(name: str, features_dict: Dict[str, torch.Tensor]):
            def hook(module, input, output):
                features_dict[name] = output

            return hook

        # Helper function to register hooks for a model
        def register_model_hooks(model, features_dict: Dict[str, torch.Tensor]):
            for name, module in model.named_modules():
                if any(layer_type in name for layer_type in ['attention', 'mlp', 'intermediate']):
                    module.register_forward_hook(create_hook(name, features_dict))

        # Register hooks for both models
        register_model_hooks(self.teacher_model, self.teacher_features)
        register_model_hooks(
            self.student_model.module if isinstance(self.student_model, DDP) else self.student_model,
            self.student_features
        )

    def black_box_distillation_loss(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute black box distillation loss using KL divergence
        """
        try:
            # Apply temperature scaling
            scaled_teacher_logits = teacher_logits / temperature
            scaled_student_logits = student_logits / temperature

            # Compute soft targets and log probabilities
            soft_targets = F.softmax(scaled_teacher_logits, dim=-1)
            log_probs = F.log_softmax(scaled_student_logits, dim=-1)

            # Compute KL divergence loss
            loss = F.kl_div(
                log_probs,
                soft_targets,
                reduction='batchmean',
                log_target=False
            ) * (temperature ** 2)

            return loss
        except Exception as e:
            logger.error(f"Error in black box distillation: {str(e)}")
            raise

    def white_box_distillation_loss(
            self,
            teacher_outputs: Any,
            student_outputs: Any,
            attention_weight: float = 0.5,
            hidden_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Compute white box distillation loss using intermediate features
        """
        try:
            total_loss = 0.0

            # 1. Hidden state matching
            def match_hidden_states(teacher_hidden, student_hidden):
                losses = []
                for t_hidden, s_hidden in zip(teacher_hidden, student_hidden):
                    if t_hidden.shape != s_hidden.shape:
                        # Adapt dimensions using interpolation
                        s_hidden = self._adapt_tensor_size(s_hidden, t_hidden.shape)
                    losses.append(F.mse_loss(s_hidden, t_hidden))
                return sum(losses) / len(losses)

            # 2. Attention pattern matching
            def match_attention_patterns(teacher_attentions, student_attentions):
                losses = []
                for t_att, s_att in zip(teacher_attentions, student_attentions):
                    if t_att.shape != s_att.shape:
                        s_att = self._adapt_attention_size(s_att, t_att.shape)
                    losses.append(F.mse_loss(s_att, t_att))
                return sum(losses) / len(losses)

            hidden_loss = match_hidden_states(
                teacher_outputs.hidden_states,
                student_outputs.hidden_states
            )

            attention_loss = match_attention_patterns(
                teacher_outputs.attentions,
                student_outputs.attentions
            )

            total_loss = (hidden_weight * hidden_loss) + (attention_weight * attention_loss)
            return total_loss

        except Exception as e:
            logger.error(f"Error in white box distillation: {str(e)}")
            raise

    def _adapt_tensor_size(self, source: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """Adapt tensor size using interpolation"""
        if len(source.shape) == 3:  # For hidden states
            return F.interpolate(
                source.unsqueeze(1),
                size=target_shape[-1],
                mode='linear'
            ).squeeze(1)
        return source

    def _adapt_attention_size(self, source: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """Adapt attention pattern size"""
        return F.interpolate(
            source.view(*source.shape[:-2], -1),
            size=target_shape[-1],
            mode='linear'
        ).view(*target_shape)

    def train(self):
        """Enhanced training loop with proper error handling and monitoring"""
        try:
            train_loader, eval_loader = self._setup_data_loaders()
            optimizer = torch.optim.AdamW(
                self.student_model.parameters(),
                lr=self.args.learning_rate
            )

            # Setup learning rate scheduler
            total_steps = len(train_loader) * self.args.num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_steps
            )

            best_eval_loss = float('inf')
            for epoch in range(self.args.num_epochs):
                train_loss = self._train_epoch(
                    epoch,
                    train_loader,
                    optimizer,
                    scheduler
                )

                eval_loss = self._evaluate(eval_loader)

                # Save best model
                if eval_loss < best_eval_loss and self.local_rank == 0:
                    best_eval_loss = eval_loss
                    self._save_model(f"{self.args.output_dir}/best_model")

                if self.local_rank == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.args.num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Eval Loss: {eval_loss:.4f}"
                    )

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        finally:
            # Clean up
            if self.local_rank == 0:
                self._save_model(f"{self.args.output_dir}/final_model")

    def _train_epoch(self, epoch: int, train_loader: DataLoader,
                     optimizer: torch.optim.Optimizer,
                     scheduler: Any) -> float:
        """Train for one epoch"""
        self.student_model.train()
        total_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}",
            disable=self.local_rank != 0
        )

        for batch_idx, batch in enumerate(progress_bar):
            try:
                loss = self._process_batch(batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                # Update progress bar
                if self.local_rank == 0 and batch_idx % self.args.logging_steps == 0:
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
                    })

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue

        return total_loss / len(train_loader)

    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a single batch and compute loss"""
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**batch)

        student_outputs = self.student_model(**batch)

        if self.args.distillation_type == "black_box":
            loss = self.black_box_distillation_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                self.args.temperature
            )
        elif self.args.distillation_type == "white_box":
            loss = self.white_box_distillation_loss(
                teacher_outputs,
                student_outputs,
                self.args.attention_weight,
                self.args.hidden_weight
            )
        else:  # combined
            loss = (
                    self.args.alpha * self.black_box_distillation_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                self.args.temperature
            ) +
                    (1 - self.args.alpha) * self.white_box_distillation_loss(
                teacher_outputs,
                student_outputs,
                self.args.attention_weight,
                self.args.hidden_weight
            )
            )

        return loss

    def _save_model(self, path: str):
        """Save the student model"""
        model_to_save = (
            self.student_model.module
            if isinstance(self.student_model, DDP)
            else self.student_model
        )
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)