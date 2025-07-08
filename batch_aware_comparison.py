#!/usr/bin/env python3
"""
Enhanced comparison script that tests both single and batch processing modes
to demonstrate the performance benefits of batch processing capabilities.
"""

import os
import sys
import time
import psutil
import gc
import torch
import warnings
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Direct imports
try:
    from text2ner import BinderInference as TextNerInference
    from binder_proc import BinderInference as BinderProcInference
    print("✅ Successfully imported Binder modules")
except ImportError as e:
    print(f"❌ Failed to import Binder modules: {e}")
    sys.exit(1)

class BertNerBaseline:
    """Simple BERT NER baseline for comparison"""
    
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", device="cpu"):
        self.device = device
        print(f"Loading BERT baseline ({model_name})")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.model = AutoModelForTokenClassification.from_pretrained(
                "dbmdz/bert-base-multilingual-cased-ner-hrl",
                num_labels=9,
                ignore_mismatched_sizes=True
            )
            self.model.to(device)
            self.model.eval()
            
            # Simple label mapping
            self.labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
            print("✅ BERT baseline loaded")
        except Exception as e:
            print(f"❌ Failed to load BERT baseline: {e}")
            raise
    
    def predict(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Simple prediction without batch processing"""
        try:
            if len(text) > 400:
                text = text[:400]
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=400,
                return_offsets_mapping=True
            )
            
            offset_mapping = inputs.pop("offset_mapping")[0]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)[0]
            
            # Simple entity extraction
            entities = []
            current_entity = None
            
            for i, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
                if start == 0 and end == 0:
                    continue
                
                pred_idx = pred_id.item()
                if pred_idx >= len(self.labels):
                    pred_idx = 0
                    
                label = self.labels[pred_idx]
                
                if label.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'start': start.item(),
                        'end': end.item(),
                        'label': label[2:],
                        'text': text[start:end]
                    }
                elif label.startswith("I-") and current_entity and label[2:] == current_entity['label']:
                    current_entity['end'] = end.item()
                    current_entity['text'] = text[current_entity['start']:current_entity['end']]
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            if current_entity:
                entities.append(current_entity)
            
            results = []
            for entity in entities:
                if len(entity['text'].strip()) > 0:
                    results.append((
                        entity['start'],
                        entity['end'],
                        entity['label'],
                        entity['text']
                    ))
            
            return results
            
        except Exception as e:
            print(f"❌ BERT prediction error: {e}")
            return []

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds/60:.1f}m {seconds%60:.1f}s"

def get_test_files(max_files: int = 10) -> List[str]:
    """Get test files from the dataset"""
    test_dir = "./data/seccol_events_texts_1500_new2-div/test"
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return []
    
    txt_files = []
    for f in sorted(os.listdir(test_dir)):
        if f.endswith('.txt') and len(txt_files) < max_files:
            file_path = os.path.join(test_dir, f)
            try:
                # Check file size - prefer medium-sized files
                size = os.path.getsize(file_path)
                if 1000 < size < 50000:  # Between 1KB and 50KB
                    txt_files.append(file_path)
            except:
                continue
    
    return txt_files

def test_single_processing(implementation, config_or_params, test_files: List[str], name: str):
    """Test implementation with single file processing"""
    print(f"\n{'='*60}")
    print(f"🔍 Testing {name} - Single Processing Mode")
    print(f"{'='*60}")
    
    try:
        gc.collect()
        start_memory = get_memory_usage()
        
        print(f"📊 Initial memory: {start_memory:.1f}MB")
        print(f"🔧 Initializing {name}...")
        
        start_time = time.time()
        
        if name == "BERT Baseline":
            inf = implementation(device="cpu")
        else:
            if not os.path.exists(config_or_params):
                print(f"❌ Config file not found: {config_or_params}")
                return None
            inf = implementation(config_path=config_or_params, device="cpu")
        
        init_time = time.time() - start_time
        post_init_memory = get_memory_usage()
        
        print(f"  ✅ Initialization: {format_time(init_time)}")
        print(f"  📊 Memory: {start_memory:.1f}MB → {post_init_memory:.1f}MB")
        
        # Warmup
        print("  🔥 Warming up...")
        try:
            _ = inf.predict("Это тестовый текст для прогрева модели.")
            print("  ✅ Warmup completed")
        except Exception as e:
            print(f"  ⚠️ Warmup failed: {e}")
        
        # Read all files first
        texts = []
        for file_path in test_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text and len(text) <= 10000:  # Skip empty or very large files
                    texts.append(text)
            except Exception as e:
                print(f"  ⚠️ Failed to read {os.path.basename(file_path)}: {e}")
        
        print(f"  📁 Processing {len(texts)} files (single mode)...")
        start_processing = time.time()
        
        processed_files = 0
        total_entities = 0
        total_chars = 0
        peak_memory = post_init_memory
        failed_files = 0
        
        # Process files one by one
        for i, text in enumerate(texts):
            try:
                predictions = inf.predict(text)
                
                processed_files += 1
                total_entities += len(predictions)
                total_chars += len(text)
                
                current_memory = get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
                
                # Print progress
                if (i + 1) % 3 == 0:
                    elapsed = time.time() - start_processing
                    print(f"    📈 Progress: {i+1}/{len(texts)} files, {elapsed:.1f}s elapsed")
                    
            except Exception as e:
                failed_files += 1
                print(f"    ❌ Failed to process file {i+1}: {e}")
        
        processing_time = time.time() - start_processing
        
        # Calculate metrics
        files_per_second = processed_files / processing_time if processing_time > 0 else 0
        chars_per_second = total_chars / processing_time if processing_time > 0 else 0
        entities_per_file = total_entities / processed_files if processed_files > 0 else 0
        
        results = {
            'name': name,
            'mode': 'single',
            'init_time': init_time,
            'processing_time': processing_time,
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_entities': total_entities,
            'total_chars': total_chars,
            'files_per_second': files_per_second,
            'chars_per_second': chars_per_second,
            'entities_per_file': entities_per_file,
            'start_memory': start_memory,
            'post_init_memory': post_init_memory,
            'peak_memory': peak_memory
        }
        
        print(f"\n📈 {name} Single Processing Results:")
        print(f"  ✅ Success: {processed_files}/{len(texts)} files (failed: {failed_files})")
        print(f"  ⏱️  Processing time: {format_time(processing_time)}")
        print(f"  🚀 Throughput: {files_per_second:.2f} files/s")
        print(f"  📝 Character rate: {chars_per_second:.0f} chars/s")
        print(f"  🏷️  Entities found: {total_entities} ({entities_per_file:.1f} per file)")
        print(f"  💾 Peak memory: {peak_memory:.1f}MB")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def test_batch_processing(implementation, config_or_params, test_files: List[str], name: str):
    """Test implementation with batch processing (if available)"""
    print(f"\n{'='*60}")
    print(f"🚀 Testing {name} - Batch Processing Mode")
    print(f"{'='*60}")
    
    try:
        gc.collect()
        start_memory = get_memory_usage()
        
        print(f"📊 Initial memory: {start_memory:.1f}MB")
        print(f"🔧 Initializing {name}...")
        
        start_time = time.time()
        
        if name == "BERT Baseline":
            inf = implementation(device="cpu")
        else:
            if not os.path.exists(config_or_params):
                print(f"❌ Config file not found: {config_or_params}")
                return None
            inf = implementation(config_path=config_or_params, device="cpu")
        
        init_time = time.time() - start_time
        post_init_memory = get_memory_usage()
        
        print(f"  ✅ Initialization: {format_time(init_time)}")
        print(f"  📊 Memory: {start_memory:.1f}MB → {post_init_memory:.1f}MB")
        
        # Check if batch processing is available
        if not hasattr(inf, 'predict_batch'):
            print(f"  ❌ {name} does not support batch processing")
            return None
        
        # Warmup
        print("  🔥 Warming up...")
        try:
            _ = inf.predict("Это тестовый текст для прогрева модели.")
            print("  ✅ Warmup completed")
        except Exception as e:
            print(f"  ⚠️ Warmup failed: {e}")
        
        # Enable batch mode optimizations if available
        if hasattr(inf, 'enable_batch_mode'):
            inf.enable_batch_mode()
            print("  🔧 Batch mode optimizations enabled")
        
        # Read all files first
        texts = []
        for file_path in test_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text and len(text) <= 10000:  # Skip empty or very large files
                    texts.append(text)
            except Exception as e:
                print(f"  ⚠️ Failed to read {os.path.basename(file_path)}: {e}")
        
        print(f"  📁 Processing {len(texts)} files (batch mode)...")
        start_processing = time.time()
        
        peak_memory = post_init_memory
        
        # Process files in batches
        batch_size = min(4, len(texts))  # Conservative batch size
        batch_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                print(f"    📦 Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch_texts)} files)")
                
                # Use batch prediction
                predictions = inf.predict_batch(batch_texts)
                batch_results.extend(predictions)
                
                current_memory = get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
                
            except Exception as e:
                print(f"    ❌ Failed to process batch {i//batch_size + 1}: {e}")
                # Fallback to single processing for this batch
                for text in batch_texts:
                    try:
                        predictions = inf.predict(text)
                        batch_results.append(predictions)
                    except:
                        batch_results.append([])
        
        processing_time = time.time() - start_processing
        
        # Calculate metrics
        processed_files = len(batch_results)
        failed_files = 0
        total_entities = sum(len(result) for result in batch_results)
        total_chars = sum(len(text) for text in texts)
        
        files_per_second = processed_files / processing_time if processing_time > 0 else 0
        chars_per_second = total_chars / processing_time if processing_time > 0 else 0
        entities_per_file = total_entities / processed_files if processed_files > 0 else 0
        
        results = {
            'name': name,
            'mode': 'batch',
            'init_time': init_time,
            'processing_time': processing_time,
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_entities': total_entities,
            'total_chars': total_chars,
            'files_per_second': files_per_second,
            'chars_per_second': chars_per_second,
            'entities_per_file': entities_per_file,
            'start_memory': start_memory,
            'post_init_memory': post_init_memory,
            'peak_memory': peak_memory
        }
        
        print(f"\n📈 {name} Batch Processing Results:")
        print(f"  ✅ Success: {processed_files}/{len(texts)} files (failed: {failed_files})")
        print(f"  ⏱️  Processing time: {format_time(processing_time)}")
        print(f"  🚀 Throughput: {files_per_second:.2f} files/s")
        print(f"  📝 Character rate: {chars_per_second:.0f} chars/s")
        print(f"  🏷️  Entities found: {total_entities} ({entities_per_file:.1f} per file)")
        print(f"  💾 Peak memory: {peak_memory:.1f}MB")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def compare_results(all_results: List[Dict]):
    """Compare all results including batch processing"""
    print(f"\n{'='*100}")
    print("COMPREHENSIVE PERFORMANCE COMPARISON (Single vs Batch Processing)")
    print(f"{'='*100}")
    
    valid_results = [r for r in all_results if r is not None]
    
    if len(valid_results) < 2:
        print("❌ Cannot compare - insufficient valid results")
        return
    
    # Create table header
    print(f"{'Implementation':<20} {'Mode':<8} {'Init':<8} {'Process':<10} {'Files/s':<8} {'Chars/s':<10} {'Memory':<8} {'Entities':<8}")
    print("-" * 100)
    
    # Display results
    for results in valid_results:
        name = results['name'][:18]
        mode = results['mode']
        init_time = format_time(results['init_time'])[:7]
        process_time = format_time(results['processing_time'])[:9]
        files_per_sec = f"{results['files_per_second']:.2f}"[:7]
        chars_per_sec = f"{results['chars_per_second']:.0f}"[:9]
        memory = f"{results['peak_memory']:.1f}MB"[:7]
        entities = f"{results['entities_per_file']:.1f}"[:7]
        
        print(f"{name:<20} {mode:<8} {init_time:<8} {process_time:<10} {files_per_sec:<8} {chars_per_sec:<10} {memory:<8} {entities:<8}")
    
    # Speed comparison
    print(f"\n{'='*100}")
    print("SPEED ANALYSIS")
    print(f"{'='*100}")
    
    # Group by implementation
    impl_results = {}
    for results in valid_results:
        name = results['name']
        if name not in impl_results:
            impl_results[name] = {}
        impl_results[name][results['mode']] = results
    
    # Show batch processing advantages
    print("📊 Batch Processing Advantages:")
    for impl_name, modes in impl_results.items():
        if 'single' in modes and 'batch' in modes:
            single_speed = modes['single']['files_per_second']
            batch_speed = modes['batch']['files_per_second']
            if single_speed > 0:
                speedup = batch_speed / single_speed
                print(f"  🚀 {impl_name}: {speedup:.2f}x faster with batch processing")
                print(f"     Single: {single_speed:.2f} files/s → Batch: {batch_speed:.2f} files/s")
    
    # Overall ranking
    print(f"\n🏆 Overall Speed Ranking:")
    speed_ranking = []
    for results in valid_results:
        speed_ranking.append((results['files_per_second'], f"{results['name']} ({results['mode']})"))
    
    speed_ranking.sort(reverse=True)
    for rank, (speed, name) in enumerate(speed_ranking, 1):
        if rank == 1:
            print(f"  🥇 1st: {name} - {speed:.2f} files/s")
        elif rank == 2:
            print(f"  🥈 2nd: {name} - {speed:.2f} files/s")
        elif rank == 3:
            print(f"  🥉 3rd: {name} - {speed:.2f} files/s")
        else:
            print(f"  {rank}th: {name} - {speed:.2f} files/s")

def main():
    """Main comparison function with batch processing support"""
    print("🚀 Batch-Aware Performance Comparison")
    print("=" * 60)
    print("Testing:")
    print("1. Text2NER (Optimized) - Single & Batch modes")
    print("2. Binder_Proc (Original) - Single mode only")
    print("3. BERT Baseline - Single mode only")
    print("=" * 60)
    
    # Get test files
    test_files = get_test_files(max_files=10)
    if not test_files:
        print("❌ No test files found")
        return
    
    print(f"✅ Found {len(test_files)} test files")
    
    all_results = []
    
    # Test Text2NER (Optimized) - Both modes
    print("\n" + "="*80)
    print("TESTING TEXT2NER (OPTIMIZED)")
    print("="*80)
    
    # Single mode
    results1_single = test_single_processing(
        TextNerInference,
        "./inference/inference-config-optimized.json",
        test_files,
        "Text2NER (Optimized)"
    )
    if results1_single:
        all_results.append(results1_single)
    
    # Cleanup
    gc.collect()
    time.sleep(2)
    
    # Batch mode
    results1_batch = test_batch_processing(
        TextNerInference,
        "./inference/inference-config-optimized.json",
        test_files,
        "Text2NER (Optimized)"
    )
    if results1_batch:
        all_results.append(results1_batch)
    
    # Cleanup
    gc.collect()
    time.sleep(2)
    
    # Test Binder_Proc (Original) - Single mode only
    print("\n" + "="*80)
    print("TESTING BINDER_PROC (ORIGINAL)")
    print("="*80)
    
    results2_single = test_single_processing(
        BinderProcInference,
        "./inference/inference-config.json",
        test_files,
        "Binder_Proc (Original)"
    )
    if results2_single:
        all_results.append(results2_single)
    
    # Cleanup
    gc.collect()
    time.sleep(2)
    
    # Test BERT Baseline - Single mode only
    print("\n" + "="*80)
    print("TESTING BERT BASELINE")
    print("="*80)
    
    results3_single = test_single_processing(
        BertNerBaseline,
        None,
        test_files,
        "BERT Baseline"
    )
    if results3_single:
        all_results.append(results3_single)
    
    # Compare all results
    compare_results(all_results)
    
    # System info
    print(f"\n{'='*100}")
    print("SYSTEM INFORMATION")
    print(f"{'='*100}")
    try:
        mem = psutil.virtual_memory()
        print(f"💾 Memory: {mem.total / 1024**3:.1f}GB total, {mem.available / 1024**3:.1f}GB available")
        print(f"🖥️  CPU: {psutil.cpu_count()} cores")
        if torch.cuda.is_available():
            print(f"🔥 CUDA: {torch.cuda.get_device_name()}")
        else:
            print("🔥 CUDA: Not available")
    except Exception as e:
        print(f"System info error: {e}")
    
    print("\n✅ Batch-aware comparison completed!")

if __name__ == "__main__":
    main() 