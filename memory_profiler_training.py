#!/usr/bin/env python3
"""
Memory profiling script for Binder training.
This script provides detailed memory usage analysis using PyTorch's profiler.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from train_binder import BinderTraining
import logging

logger = logging.getLogger(__name__)

class MemoryProfiledBinderTraining(BinderTraining):
    """
    Extended BinderTraining class with memory profiling capabilities.
    """
    
    def __init__(self, config_path: str, device: str = "auto", profile_memory: bool = True):
        super().__init__(config_path, device)
        self.profile_memory = profile_memory
        self.profiler = None
        self.memory_snapshots = []
        
    def setup_profiler(self, output_dir: str = None):
        """Setup PyTorch profiler for memory analysis."""
        if not self.profile_memory:
            logger.info("Memory profiling disabled")
            return
            
        if output_dir is None:
            output_dir = self.training_args.output_dir if hasattr(self, 'training_args') else "./profiler_output"
        
        # Always create the profiler output directory
        self.profiler_output_dir = os.path.join(output_dir, "profiler_output")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.profiler_output_dir, exist_ok=True)
        
        # Configure profiler based on available hardware
        if torch.cuda.is_available():
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            logger.info("📊 CUDA available - enabling CPU and GPU profiling")
        else:
            activities = [ProfilerActivity.CPU]
            logger.info("📊 CUDA not available - enabling CPU-only profiling")
        
        try:
            # Configure profiler with more frequent sampling for short training runs
            self.profiler = profile(
                activities=activities,
                schedule=schedule(wait=0, warmup=1, active=2, repeat=1),  # More frequent sampling
                on_trace_ready=self._trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            logger.info(f"📊 Memory profiler initialized. Output will be saved to: {self.profiler_output_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize PyTorch profiler: {e}. Will use basic memory logging only.")
            self.profiler = None
        
        # Always log initial memory state
        self._log_memory_state("initialization")
        
    def _log_memory_state(self, stage: str):
        """Log current memory state to a file."""
        if not hasattr(self, 'profiler_output_dir'):
            return
            
        memory_info = {}
        
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            memory_info['gpu'] = {
                'allocated_gb': torch.cuda.memory_allocated(device_idx) / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved(device_idx) / (1024**3),
                'device': device_idx
            }
        
        try:
            import psutil
            process = psutil.Process()
            memory_info['cpu'] = {
                'rss_gb': process.memory_info().rss / (1024**3),
                'vms_gb': process.memory_info().vms / (1024**3),
                'percent': process.memory_percent()
            }
        except ImportError:
            memory_info['cpu'] = {'status': 'psutil not available'}
        
        # Log to file
        log_file = os.path.join(self.profiler_output_dir, "memory_log.txt")
        with open(log_file, "a") as f:
            f.write(f"\n=== Memory State at {stage} ===\n")
            if 'gpu' in memory_info:
                f.write(f"GPU (device {memory_info['gpu']['device']}): "
                       f"Allocated={memory_info['gpu']['allocated_gb']:.2f}GB, "
                       f"Reserved={memory_info['gpu']['reserved_gb']:.2f}GB\n")
            if 'cpu' in memory_info:
                if 'rss_gb' in memory_info['cpu']:
                    f.write(f"CPU: RSS={memory_info['cpu']['rss_gb']:.2f}GB, "
                           f"VMS={memory_info['cpu']['vms_gb']:.2f}GB, "
                           f"Percent={memory_info['cpu']['percent']:.1f}%\n")
                else:
                    f.write(f"CPU: {memory_info['cpu']['status']}\n")
        
        # Also log to console
        if 'gpu' in memory_info:
            logger.info(f"📊 Memory at {stage} - GPU: {memory_info['gpu']['allocated_gb']:.2f}GB allocated, "
                       f"{memory_info['gpu']['reserved_gb']:.2f}GB reserved")
        if 'cpu' in memory_info and 'rss_gb' in memory_info['cpu']:
            logger.info(f"📊 Memory at {stage} - CPU: {memory_info['cpu']['rss_gb']:.2f}GB RSS")

    def _trace_handler(self, prof):
        """Handle profiler traces and save memory analysis."""
        if not hasattr(self, 'profiler_output_dir'):
            return
            
        # Save detailed memory usage table
        memory_table = prof.key_averages().table(
            sort_by="cuda_memory_usage", 
            row_limit=50
        )
        
        step_num = len(self.memory_snapshots)
        
        # Write memory table to file
        with open(os.path.join(self.profiler_output_dir, f"memory_analysis_step_{step_num}.txt"), "w") as f:
            f.write(f"Memory Analysis - Step {step_num}\n")
            f.write("=" * 50 + "\n")
            f.write(memory_table)
            f.write("\n\n")
            
            # Add current GPU memory stats
            if torch.cuda.is_available():
                device_idx = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
                reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
                f.write(f"Current GPU Memory:\n")
                f.write(f"  Allocated: {allocated:.2f} GB\n")
                f.write(f"  Reserved: {reserved:.2f} GB\n")
        
        # Save Chrome trace for visualization
        prof.export_chrome_trace(os.path.join(self.profiler_output_dir, f"trace_step_{step_num}.json"))
        
        # Store snapshot for summary
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            snapshot = {
                'step': step_num,
                'allocated_gb': torch.cuda.memory_allocated(device_idx) / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved(device_idx) / (1024**3),
                'top_operations': prof.key_averages(group_by_input_shape=True)[:10]
            }
            self.memory_snapshots.append(snapshot)
        
        logger.info(f"📊 Memory profile saved for step {step_num}")
        
    def train_with_profiling(self, do_eval: bool = None, do_predict: bool = None):
        """Train with memory profiling enabled."""
        # Setup first to get training_args
        self.setup()
        
        # Temporarily disable profiling during dataset preprocessing to avoid serialization issues
        original_profile_memory = self.profile_memory
        self.profile_memory = False
        
        try:
            # Run setup and dataset preprocessing without profiling
            self._setup_tokenizer_and_entity_types()
            self.prepare_train_dataset()
            
            if do_eval is None:
                do_eval = self.training_args.do_eval
            if do_predict is None:
                do_predict = self.training_args.do_predict
                
            if do_eval:
                self.prepare_validation_dataset()
            if do_predict:
                self.prepare_test_dataset()
            
            # Re-enable profiling for actual training
            self.profile_memory = original_profile_memory
            
            # Now setup profiler and run training with profiling
            self.setup_profiler()
            
            if self.profiler is not None:
                self.profiler.start()
            
            # Run the actual training part (without dataset preparation)
            trainer = self._train_with_prepared_data(do_eval=do_eval, do_predict=do_predict)
             
            # Log memory state after training
            self._log_memory_state("training_completed")
            
            # Generate final memory summary
            self._generate_memory_summary()
            
            return trainer
            
        finally:
            # Always log final memory state and generate summary
            self._log_memory_state("cleanup")
            if self.profiler is not None:
                self.profiler.stop()
            # Ensure summary is generated even if profiler failed
            if hasattr(self, 'profiler_output_dir'):
                self._generate_basic_summary()
                
    def _train_with_prepared_data(self, do_eval: bool = None, do_predict: bool = None):
        """Run training with already prepared datasets."""
        # Load model config
        from src.config import BinderConfig
        from src.model import Binder
        
        config = BinderConfig(
            pretrained_model_name_or_path=self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
            hidden_dropout_prob=self.model_args.hidden_dropout_prob,
            max_span_width=self.data_args.max_seq_length + 1,
            use_span_width_embedding=self.model_args.use_span_width_embedding,
            linear_size=self.model_args.linear_size,
            init_temperature=self.model_args.init_temperature,
            start_loss_weight=self.model_args.start_loss_weight,
            end_loss_weight=self.model_args.end_loss_weight,
            span_loss_weight=self.model_args.span_loss_weight,
            threshold_loss_weight=self.model_args.threshold_loss_weight,
            ner_loss_weight=self.model_args.ner_loss_weight,
        )

        # Load model
        logger.info(f"Initializing Binder model with {len(self.entity_type_id_to_str)} entity types")
        # If a Binder checkpoint path is supplied, load weights from there to continue training
        if getattr(self.model_args, "binder_model_name_or_path", None):
            try:
                logger.info(
                    f"Loading Binder weights from checkpoint: {self.model_args.binder_model_name_or_path}"
                )
                self.model = Binder.from_pretrained(
                    self.model_args.binder_model_name_or_path,
                    config=config,
                    ignore_mismatched_sizes=True,
                )
                logger.info("✅ Loaded Binder checkpoint successfully – continuing training from previous weights.")
            except Exception as e:
                logger.warning(
                    f"⚠️  Could not load Binder checkpoint due to: {e}. Initializing a new Binder model instead."
                )
                self.model = Binder(config)
        else:
            # No checkpoint provided → start from scratch
            self.model = Binder(config)
        
        # Validate model initialization
        logger.info("Model initialized successfully")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())} total parameters")
        logger.info(f"Model trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters")
        
        # Set up post-processing function
        from src import utils as postprocess_utils
        def post_processing_function(examples, features, predictions, stage="eval"):
            return postprocess_utils.postprocess_nested_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                id_to_type=self.entity_type_id_to_str,
                max_span_length=self.data_args.max_span_length,
                output_dir=self.training_args.output_dir if self.training_args.should_save else None,
                log_level=self.training_args.get_process_log_level(),
                prefix=stage,
                tokenizer=self.tokenizer,
                train_file=self.data_args.train_file,
            )

        # Initialize Trainer
        from transformers import EarlyStoppingCallback
        from src.trainer import BinderTrainer
        from src.memory_callback import MemoryUsageCallback
        
        # Only use EarlyStoppingCallback when doing evaluation, since it requires load_best_model_at_end=True
        callbacks = [MemoryUsageCallback()]
        if do_eval and self.eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=20))
        
        # Disable evaluation in training arguments if do_eval=False
        if not do_eval:
            # Override evaluation settings to prevent trainer from trying to evaluate
            self.training_args.evaluation_strategy = "no"
            self.training_args.eval_steps = None
            self.training_args.eval_delay = 0
            self.training_args.load_best_model_at_end = False
            self.training_args.metric_for_best_model = None
            logger.info("🚫 Evaluation disabled - overriding training arguments to prevent evaluation during training")
        
        trainer = BinderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset if do_eval else None,
            eval_examples=self.eval_examples if do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=post_processing_function,
            callbacks=callbacks,
            compute_metrics=None,
        )

        # Continue with the rest of training logic from parent class
        return self._run_training_loop(trainer, do_eval, do_predict)
        
    def _run_training_loop(self, trainer, do_eval, do_predict):
        """Execute the training loop with memory logging."""
        from transformers.trainer_utils import get_last_checkpoint
        import os
        
        # --------------------------------------------------
        # Memory usage logging BEFORE training begins
        # --------------------------------------------------
        try:
            device = self.training_args.device
        except AttributeError:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available() and device.type == "cuda":
            # Ensure we look at the correct device (in multi-GPU setups)
            gpu_index = device.index if device.index is not None else torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_index) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(gpu_index) / (1024 ** 3)
            logger.info(
                f"📊 GPU memory usage before training – allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB (device {gpu_index})"
            )
        else:
            try:
                import psutil
                process_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
                logger.info(f"📊 CPU memory (RSS) before training: {process_mem:.2f} GB")
            except ImportError:
                logger.info("📊 psutil not installed; CPU memory usage unavailable")

        # Training (always performed)
        logger.info("*** Starting Training ***")
        checkpoint = None
        # Only resume if user explicitly provided checkpoint path
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        # If user did not provide checkpoint but wants to continue, detect last checkpoint only when overwrite_output_dir is False
        elif not self.training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(self.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # Save model as safetensors for inference
        safetensors_dir = os.path.join(self.training_args.output_dir, "safetensors_model")
        self.save_model_as_safetensors(safetensors_dir, save_tokenizer=True)
        
        logger.info("*** Training completed successfully ***")

        # Optional evaluation
        if do_eval and self.eval_dataset is not None:
            logger.info("*** Starting Evaluation ***")
            eval_metrics = trainer.evaluate()
            
            max_eval_samples = self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(self.eval_dataset)
            eval_metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))

            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
            logger.info("*** Evaluation completed ***")

        # Optional prediction
        if do_predict and self.predict_dataset is not None:
            logger.info("*** Starting Prediction ***")
            results = trainer.predict(self.predict_dataset, self.predict_examples)
            predict_metrics = results.metrics
            
            max_predict_samples = (
                self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(self.predict_dataset)
            )
            predict_metrics["predict_samples"] = min(max_predict_samples, len(self.predict_dataset))

            trainer.log_metrics("predict", predict_metrics)
            trainer.save_metrics("predict", predict_metrics)
            logger.info("*** Prediction completed ***")

        return trainer
                 
    def _generate_memory_summary(self):
        """Generate a comprehensive memory usage summary."""
        if not self.memory_snapshots:
            logger.info("No memory snapshots available for summary")
            return
            
        output_dir = os.path.join(self.training_args.output_dir, "profiler_output")
        summary_file = os.path.join(output_dir, "memory_summary.txt")
        
        with open(summary_file, "w") as f:
            f.write("MEMORY USAGE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            allocated_values = [s['allocated_gb'] for s in self.memory_snapshots]
            reserved_values = [s['reserved_gb'] for s in self.memory_snapshots]
            
            f.write(f"Memory Statistics:\n")
            f.write(f"  Peak Allocated: {max(allocated_values):.2f} GB\n")
            f.write(f"  Peak Reserved: {max(reserved_values):.2f} GB\n")
            f.write(f"  Average Allocated: {sum(allocated_values)/len(allocated_values):.2f} GB\n")
            f.write(f"  Average Reserved: {sum(reserved_values)/len(reserved_values):.2f} GB\n")
            f.write(f"  Total Snapshots: {len(self.memory_snapshots)}\n\n")
            
            # Step-by-step breakdown
            f.write("Step-by-Step Memory Usage:\n")
            f.write("-" * 30 + "\n")
            for snapshot in self.memory_snapshots:
                f.write(f"Step {snapshot['step']}: "
                       f"Allocated={snapshot['allocated_gb']:.2f}GB, "
                       f"Reserved={snapshot['reserved_gb']:.2f}GB\n")
                       
        logger.info(f"📈 Memory summary saved to: {summary_file}")
        
        # Print summary to console
        allocated_values = [s['allocated_gb'] for s in self.memory_snapshots]
        reserved_values = [s['reserved_gb'] for s in self.memory_snapshots]
        
        logger.info("🔍 MEMORY USAGE SUMMARY:")
        logger.info(f"   Peak Allocated: {max(allocated_values):.2f} GB")
        logger.info(f"   Peak Reserved: {max(reserved_values):.2f} GB")
        logger.info(f"   Average Allocated: {sum(allocated_values)/len(allocated_values):.2f} GB")
        logger.info(f"   Average Reserved: {sum(reserved_values)/len(reserved_values):.2f} GB")

    def _generate_basic_summary(self):
        """Generate a basic memory summary even if profiler didn't work properly."""
        if not hasattr(self, 'profiler_output_dir'):
            return
            
        summary_file = os.path.join(self.profiler_output_dir, "basic_memory_summary.txt")
        
        with open(summary_file, "w") as f:
            f.write("BASIC MEMORY SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Check if memory log exists
            log_file = os.path.join(self.profiler_output_dir, "memory_log.txt")
            if os.path.exists(log_file):
                f.write("Memory log file created successfully.\n")
                with open(log_file, "r") as log_f:
                    content = log_f.read()
                    f.write("\nMemory Log Contents:\n")
                    f.write("-" * 30 + "\n")
                    f.write(content)
            else:
                f.write("No memory log file found.\n")
            
            # Current memory state
            f.write("\nCurrent Memory State:\n")
            f.write("-" * 30 + "\n")
            
            if torch.cuda.is_available():
                device_idx = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
                reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
                f.write(f"GPU: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB\n")
            
            try:
                import psutil
                process = psutil.Process()
                rss = process.memory_info().rss / (1024**3)
                f.write(f"CPU: RSS={rss:.2f}GB\n")
            except ImportError:
                f.write("CPU: psutil not available\n")
        
        logger.info(f"📈 Basic memory summary saved to: {summary_file}")

def main():
    """Main function to run memory profiled training."""
    parser = argparse.ArgumentParser(description="Memory profiled Binder training")
    parser.add_argument("--config", type=str, required=True, help="Path to training config JSON")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                       help="Device to use for training")
    parser.add_argument("--no-profiling", action="store_true", 
                       help="Disable memory profiling (use regular training)")
    parser.add_argument("--train-only", action="store_true", 
                       help="Only train, skip evaluation and prediction")
    parser.add_argument("--train-and-eval", action="store_true", 
                       help="Train and evaluate")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
        
    print(f"🚀 Starting memory profiled Binder training")
    print(f"📋 Config: {args.config}")
    print(f"🖥️  Device: {args.device}")
    print(f"📊 Profiling: {'Disabled' if args.no_profiling else 'Enabled'}")
    
    try:
        # Create profiled trainer
        trainer = MemoryProfiledBinderTraining(
            config_path=args.config,
            device=args.device,
            profile_memory=not args.no_profiling
        )
        
        # Choose training mode
        if args.train_only:
            print("🎯 Mode: Training only")
            result = trainer.train_with_profiling(do_eval=False, do_predict=False)
        elif args.train_and_eval:
            print("🎯 Mode: Training + Evaluation")
            result = trainer.train_with_profiling(do_eval=True, do_predict=False)
        else:
            print("🎯 Mode: Full training (train + eval + predict)")
            result = trainer.train_with_profiling(do_eval=True, do_predict=True)
            
        print("✅ Training completed successfully!")
        
        if not args.no_profiling:
            output_dir = os.path.join(trainer.training_args.output_dir, "profiler_output")
            print(f"📊 Memory profiling results saved to: {output_dir}")
            print("💡 View Chrome traces by opening trace_step_*.json files in chrome://tracing/")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.error("Training failed", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 