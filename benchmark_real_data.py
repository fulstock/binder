#!/usr/bin/env python3
"""
Comprehensive benchmark for BinderInference using real seccol_events_texts data.
This script investigates batch vs single prediction performance and tests on real data.
"""

import json
import time
import psutil
import torch
import statistics
import threading
import random
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

def load_real_data(file_path, max_samples=20):
    """Load real data from JSON file with conservative sample limits"""
    print(f"Loading real data from: {file_path}")
    
    # Check available memory and adjust sample size accordingly
    available_memory = psutil.virtual_memory().available / (1024**3)
    if available_memory < 8:
        max_samples = min(max_samples, 10)
        print(f"⚠️  Low memory detected ({available_memory:.1f}GB). Reducing samples to {max_samples}")
    elif available_memory < 16:
        max_samples = min(max_samples, 15)
        print(f"⚠️  Limited memory detected ({available_memory:.1f}GB). Reducing samples to {max_samples}")
    
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
        print(f"Sample text lengths: {[len(item['text']) for item in data[:5]]}")
        return data
    
    except Exception as e:
        print(f"❌ Failed to load real data: {e}")
        return []

def measure_memory():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def analyze_batch_issue(inf, texts, verbose=True):
    """Detailed analysis of why batch might be slower than individual predictions"""
    if verbose:
        print("\n" + "="*60)
        print("BATCH PERFORMANCE ANALYSIS")
        print("="*60)
    
    # Test various batch sizes - conservative settings
    max_batch_size = min(len(texts), 8)  # Limit maximum batch size to 8 to prevent memory issues
    batch_sizes = [1, 2, min(5, max_batch_size), max_batch_size]
    batch_sizes = [b for b in batch_sizes if b <= len(texts)]  # Remove duplicates and invalid sizes
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(texts):
            continue
            
        test_texts = texts[:batch_size]
        
        # Individual predictions
        individual_times = []
        for text in test_texts:
            try:
                start = time.time()
                _ = run_with_timeout(lambda: inf.predict(text), 30)
                individual_times.append(time.time() - start)
            except Exception as e:
                if verbose:
                    print(f"⚠️  Individual prediction failed: {e}")
                continue
        
        # Batch prediction
        batch_time = None
        if hasattr(inf, 'predict_batch') and individual_times:
            try:
                start = time.time()
                _ = run_with_timeout(lambda: inf.predict_batch(test_texts), 60)
                batch_time = time.time() - start
            except Exception as e:
                if verbose:
                    print(f"⚠️  Batch prediction failed: {e}")
        
        if individual_times and batch_time:
            total_individual = sum(individual_times)
            speedup = total_individual / batch_time
            results[batch_size] = {
                'individual_total': total_individual,
                'batch_time': batch_time,
                'speedup': speedup,
                'individual_avg': statistics.mean(individual_times)
            }
            
            if verbose:
                print(f"\nBatch size {batch_size}:")
                print(f"  Individual total: {total_individual:.3f}s")
                print(f"  Batch total: {batch_time:.3f}s")
                print(f"  Speedup: {speedup:.2f}x {'(SLOWER!)' if speedup < 1 else '(faster)'}")
    
    return results

def comprehensive_benchmark(config_path, real_data, synthetic_data):
    """Run comprehensive benchmark with both real and synthetic data"""
    print(f"\n" + "="*60)
    print(f"COMPREHENSIVE BENCHMARK: {config_path}")
    print("="*60)
    
    # Memory tracking
    memory_before = measure_memory()
    
    # Initialize with timeout and resource limits
    print("Initializing BinderInference with resource limits...")
    try:
        # Use conservative memory limit and CPU-only mode if needed
        available_memory = psutil.virtual_memory().available / (1024**3)
        device = "cpu" if available_memory < 4 else "auto"  # Force CPU mode on very low memory systems
        
        inf = run_with_timeout(
            lambda: BinderInference(
                config_path=config_path, 
                max_memory_gb=16,  # 16GB memory limit as requested
                device=device
            ), 
            300
        )
        print("✅ BinderInference initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None
    
    memory_after_init = measure_memory()
    print(f"Memory usage: {memory_before:.1f}MB → {memory_after_init:.1f}MB (+{memory_after_init - memory_before:.1f}MB)")
    
    results = {
        'config': config_path,
        'memory_usage': memory_after_init - memory_before,
        'real_data': {},
        'synthetic_data': {},
        'batch_analysis': {}
    }
    
    # Test on real data
    if real_data:
        print(f"\n--- Testing on {len(real_data)} real texts ---")
        
        # Select subset for testing - use smaller subset to prevent resource exhaustion
        test_size = min(6, len(real_data))  # Reduced from 10 to 6 for resource efficiency
        test_real = random.sample(real_data, test_size)
        real_texts = [item['text'] for item in test_real]
        
        # Individual predictions on real data
        real_times = []
        real_entities_found = []
        for i, item in enumerate(test_real):
            print(f"  Real text {i+1}/{len(test_real)} (len={len(item['text'])}, expected={item['expected_entities']})")
            try:
                start = time.time()
                predictions = run_with_timeout(lambda: inf.predict(item['text']), 45)
                pred_time = time.time() - start
                real_times.append(pred_time)
                real_entities_found.append(len(predictions) if predictions else 0)
                print(f"    ✅ {pred_time:.3f}s, found {len(predictions) if predictions else 0} entities")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        
        if real_times:
            results['real_data'] = {
                'avg_time': statistics.mean(real_times),
                'total_time': sum(real_times),
                'throughput': len(real_times) / sum(real_times),
                'avg_entities_found': statistics.mean(real_entities_found),
                'successful_predictions': len(real_times)
            }
        
        # Batch analysis on real data - use smaller subset
        if len(real_texts) >= 2:
            print("\n--- Batch Analysis on Real Data ---")
            batch_subset = real_texts[:min(4, len(real_texts))]  # Reduced from 5 to 4
            results['batch_analysis']['real'] = analyze_batch_issue(inf, batch_subset)
    
    # Test on synthetic data
    if synthetic_data:
        print(f"\n--- Testing on {len(synthetic_data)} synthetic texts ---")
        
        # Individual predictions on synthetic data
        synthetic_times = []
        synthetic_entities_found = []
        for i, text in enumerate(synthetic_data):
            print(f"  Synthetic text {i+1}/{len(synthetic_data)} (len={len(text)})")
            try:
                start = time.time()
                predictions = run_with_timeout(lambda: inf.predict(text), 30)
                pred_time = time.time() - start
                synthetic_times.append(pred_time)
                synthetic_entities_found.append(len(predictions) if predictions else 0)
                print(f"    ✅ {pred_time:.3f}s, found {len(predictions) if predictions else 0} entities")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        
        if synthetic_times:
            results['synthetic_data'] = {
                'avg_time': statistics.mean(synthetic_times),
                'total_time': sum(synthetic_times),
                'throughput': len(synthetic_times) / sum(synthetic_times),
                'avg_entities_found': statistics.mean(synthetic_entities_found),
                'successful_predictions': len(synthetic_times)
            }
        
        # Batch analysis on synthetic data - limited subset
        if len(synthetic_data) >= 2:
            print("\n--- Batch Analysis on Synthetic Data ---")
            synthetic_subset = synthetic_data[:min(4, len(synthetic_data))]  # Limit to 4 samples
            results['batch_analysis']['synthetic'] = analyze_batch_issue(inf, synthetic_subset)
    
    # Memory cleanup
    try:
        if hasattr(inf, 'clear_cache'):
            inf.clear_cache()
    except Exception as e:
        print(f"⚠️  Cache cleanup failed: {e}")
    
    return results

def main():
    print("="*70)
    print("COMPREHENSIVE BINDER INFERENCE BENCHMARK")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Memory available: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    
    # Load real data with conservative limits
    real_data = load_real_data("data/seccol_events_texts_1500_new2-binder-div/test.json", max_samples=12)
    
    # Prepare synthetic test data - reduced set for resource efficiency
    synthetic_data = [
        "Хакеры атаковали сервер компании Microsoft.",
        "В результате кибератаки была нарушена работа критической информационной инфраструктуры.",
        "Специалисты по информационной безопасности обнаружили новую программу-вымогатель.",
        "Группировка APT29 использовала уязвимость CVE-2024-12345 для атаки на правительственные сети."
    ]  # Reduced from 5 to 4 samples
    
    print(f"\nDataset summary:")
    print(f"  Real texts: {len(real_data)}")
    print(f"  Synthetic texts: {len(synthetic_data)}")
    
    # Test original configuration
    print("\n" + "="*70)
    print("TESTING ORIGINAL CONFIGURATION")
    print("="*70)
    
    original_results = comprehensive_benchmark(
        "./inference/inference-config.json",
        real_data,
        synthetic_data
    )
    
    # Test optimized configuration
    print("\n" + "="*70)
    print("TESTING OPTIMIZED CONFIGURATION")
    print("="*70)
    
    optimized_results = comprehensive_benchmark(
        "./inference/inference-config-optimized.json",
        real_data,
        synthetic_data
    )
    
    # Compare results
    if original_results and optimized_results:
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        
        for data_type in ['real_data', 'synthetic_data']:
            if data_type in original_results and data_type in optimized_results:
                orig = original_results[data_type]
                opt = optimized_results[data_type]
                
                if orig and opt:
                    speedup = orig['avg_time'] / opt['avg_time']
                    throughput_improvement = opt['throughput'] / orig['throughput']
                    
                    print(f"\n{data_type.replace('_', ' ').title()}:")
                    print(f"  Average time: {orig['avg_time']:.3f}s → {opt['avg_time']:.3f}s")
                    print(f"  Speedup: {speedup:.2f}x")
                    print(f"  Throughput: {orig['throughput']:.2f} → {opt['throughput']:.2f} texts/sec")
                    print(f"  Throughput improvement: {throughput_improvement:.2f}x")
                    print(f"  Entities found: {orig['avg_entities_found']:.1f} → {opt['avg_entities_found']:.1f} avg")
        
        # Batch analysis summary
        print(f"\nBatch Processing Analysis:")
        for config_name, results in [("Original", original_results), ("Optimized", optimized_results)]:
            if 'batch_analysis' in results:
                print(f"\n{config_name} Config:")
                for data_type, analysis in results['batch_analysis'].items():
                    if analysis:
                        print(f"  {data_type.title()} data batch performance:")
                        for batch_size, metrics in analysis.items():
                            speedup = metrics['speedup']
                            status = "GOOD" if speedup > 1 else "POOR"
                            print(f"    Batch size {batch_size}: {speedup:.2f}x speedup [{status}]")
        
        print(f"\nMemory usage:")
        print(f"  Original: {original_results['memory_usage']:.1f}MB")
        print(f"  Optimized: {optimized_results['memory_usage']:.1f}MB")
        
    print("\n" + "="*70)
    print("BATCH PERFORMANCE RECOMMENDATIONS")
    print("="*70)
    print("• Batch processing should be faster than individual predictions")
    print("• If batch is slower, possible causes:")
    print("  - Inefficient batch implementation")
    print("  - Memory constraints causing swapping")
    print("  - Overhead not amortized over small batches")
    print("  - Different tokenization/processing paths")
    print("• Recommended batch sizes: 5-20 texts")
    print("• Consider implementing streaming batch processing")

if __name__ == "__main__":
    main() 