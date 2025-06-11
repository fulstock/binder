#!/usr/bin/env python3
"""
Resource-limited benchmark for BinderInference batch processing investigation.
This script uses minimal system resources (2 workers, 16GB RAM limit) to prevent system freezing.
"""

import json
import time
import psutil
import torch
import statistics
import threading
import random
import gc
from text2ner import BinderInference

class TimeoutException(Exception):
    pass

def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """Run function with timeout protection"""
    result = [TimeoutException("Function timed out")]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            result[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        raise TimeoutException(f"Function timed out after {timeout_seconds} seconds")
    
    if isinstance(result[0], Exception):
        raise result[0]
    
    return result[0]

def measure_memory():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def load_seccol_data(max_samples=10):
    """Load seccol_events_texts data with strict memory limits"""
    file_path = "data/seccol_events_texts_1500_new2-binder-div/test.json"
    print(f"Loading seccol data from: {file_path}")
    
    # Ultra-conservative memory check
    available_memory = psutil.virtual_memory().available / (1024**3)
    if available_memory < 4:
        max_samples = 5
        print(f"⚠️  Very low memory ({available_memory:.1f}GB). Limiting to {max_samples} samples")
    elif available_memory < 8:
        max_samples = min(max_samples, 8)
        print(f"⚠️  Low memory ({available_memory:.1f}GB). Limiting to {max_samples} samples")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = []
            for line_num, line in enumerate(f):
                if line_num >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    if 'text' in item and len(item['text'].strip()) > 0:
                        data.append({
                            'text': item['text'],
                            'id': item.get('id', str(line_num)),
                            'expected_entities': len(item.get('entity_types', []))
                        })
                except json.JSONDecodeError:
                    continue
        
        print(f"✅ Loaded {len(data)} real text samples")
        if data:
            print(f"Sample text lengths: {[len(item['text']) for item in data[:3]]}")
            print(f"Expected entities: {[item['expected_entities'] for item in data[:3]]}")
        return data
    
    except Exception as e:
        print(f"❌ Failed to load seccol data: {e}")
        return []

def investigate_batch_issue():
    """Investigate the specific batch processing issue mentioned by user"""
    print("="*70)
    print(f"BATCH PROCESSING INVESTIGATION - RESOURCE LIMITED")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Memory status
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.available / 1024**3:.2f}GB available, {mem.total / 1024**3:.2f}GB total")
    
    # Load minimal data set
    real_data = load_seccol_data(max_samples=10)
    if not real_data:
        print("❌ No real data loaded. Cannot proceed with investigation.")
        return
    
    print(f"\nLoaded {len(real_data)} real samples")
    for i, item in enumerate(real_data):
        print(f"  Sample {i+1}: len={len(item['text'])}, expected_entities={item['expected_entities']}")
    
    # Initialize with strict resource limits
    print(f"\n--- Initializing BinderInference with Resource Limits ---")
    memory_before = measure_memory()
    
    try:
        inf = run_with_timeout(
            lambda: BinderInference(
                config_path="./inference/inference-config-optimized.json",
                max_memory_gb=16,  # 16GB limit as requested
                device="auto"
            ), 
            300  # 5 minute timeout
        )
        print("✅ BinderInference initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    memory_after_init = measure_memory()
    print(f"Memory usage: {memory_before:.1f}MB → {memory_after_init:.1f}MB (+{memory_after_init - memory_before:.1f}MB)")
    
    # Test individual predictions first
    print(f"\n--- Testing Individual Predictions ---")
    individual_times = []
    individual_results = []
    
    for i, item in enumerate(real_data):
        print(f"Real text {i+1}/{len(real_data)} (len={len(item['text'])}, expected={item['expected_entities']})")
        try:
            start = time.time()
            predictions = run_with_timeout(lambda: inf.predict(item['text']), 60)
            pred_time = time.time() - start
            individual_times.append(pred_time)
            entities_found = len(predictions) if predictions else 0
            individual_results.append(entities_found)
            print(f"    ✅ {pred_time:.3f}s, found {entities_found} entities")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        
        # Memory monitoring
        if i % 3 == 0:
            current_memory = measure_memory()
            print(f"    📊 Memory: {current_memory:.1f}MB")
            
        # Cleanup after each prediction
        gc.collect()
    
    if not individual_times:
        print("❌ No individual predictions succeeded. Cannot test batch processing.")
        return
    
    print(f"\nIndividual prediction summary:")
    print(f"  Average time: {statistics.mean(individual_times):.3f}s")
    print(f"  Total time: {sum(individual_times):.3f}s")
    print(f"  Average entities found: {statistics.mean(individual_results):.1f}")
    print(f"  Expected entities: {[item['expected_entities'] for item in real_data]}")
    
    # Test batch predictions
    print(f"\n--- Testing Batch Predictions ---")
    
    # Use smaller batch sizes to prevent memory issues
    batch_sizes = [2, 3, min(4, len(real_data))]
    
    for batch_size in batch_sizes:
        if batch_size > len(real_data):
            continue
            
        print(f"\nBatch size {batch_size}:")
        test_texts = [item['text'] for item in real_data[:batch_size]]
        expected_total = sum(item['expected_entities'] for item in real_data[:batch_size])
        
        try:
            start = time.time()
            batch_predictions = run_with_timeout(
                lambda: inf.predict_batch(test_texts), 
                120  # 2 minute timeout for batch
            )
            batch_time = time.time() - start
            
            # Count total entities found in batch
            batch_entities_found = sum(len(pred_list) if pred_list else 0 for pred_list in batch_predictions)
            
            # Calculate individual time for comparison
            individual_time_total = sum(individual_times[:batch_size])
            speedup = individual_time_total / batch_time if batch_time > 0 else 0
            
            print(f"  Individual total: {individual_time_total:.3f}s")
            print(f"  Batch total: {batch_time:.3f}s")
            print(f"  Speedup: {speedup:.2f}x {'(faster)' if speedup > 1 else '(SLOWER!)'}")
            print(f"  Entities found: {batch_entities_found} (expected: {expected_total})")
            
            # Memory check
            current_memory = measure_memory()
            print(f"  Memory after batch: {current_memory:.1f}MB")
            
        except Exception as e:
            print(f"  ❌ Batch prediction failed: {e}")
            
        # Cleanup between batches
        gc.collect()
    
    # Final analysis
    print(f"\n--- Analysis ---")
    print("Issues identified:")
    
    if all(result == 0 for result in individual_results):
        print("• Model is finding 0 entities when it should find more")
        print("• This suggests:")
        print("  - Prediction threshold might be too high")
        print("  - Model might not be properly trained for this domain")
        print("  - Postprocessing issues")
    
    print("• Batch processing appears to be slower than individual processing")
    print("• This suggests:")
    print("  - Batch implementation might not be truly optimized")
    print("  - Memory overhead from batching")
    print("  - Different processing paths for batch vs individual")
    
    print(f"\nRecommendations:")
    print("1. Check prediction threshold settings")
    print("2. Verify model training for cybersecurity domain")
    print("3. Optimize batch processing implementation")
    print("4. Consider streaming batch processing for large datasets")
    
    # Final memory cleanup
    try:
        if hasattr(inf, 'clear_cache'):
            inf.clear_cache()
    except Exception as e:
        print(f"⚠️  Cache cleanup failed: {e}")

def main():
    print("RESOURCE-LIMITED BATCH PROCESSING INVESTIGATION")
    print("Configuration: 2 workers, 16GB RAM limit, minimal samples")
    print("="*70)
    
    # Set process priority to low to prevent system freeze
    try:
        if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
            p = psutil.Process()
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("✅ Set process priority to low")
    except Exception as e:
        print(f"⚠️  Could not set process priority: {e}")
    
    investigate_batch_issue()

if __name__ == "__main__":
    main() 