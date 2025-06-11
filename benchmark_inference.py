#!/usr/bin/env python3
"""
Performance benchmark for BinderInference optimization.
This script compares the original vs optimized inference performance.
"""

import time
import psutil
import torch
import statistics
import threading
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
        # Thread didn't finish in time
        raise TimeoutException(f"Function timed out after {timeout_seconds} seconds")
    
    if isinstance(result[0], Exception):
        raise result[0]
    
    return result[0]

def measure_memory():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def benchmark_inference(config_path, texts, num_runs=3, warmup=True):
    """Benchmark inference performance with timeout protection"""
    print(f"\n=== Benchmarking with config: {config_path} ===")
    
    # Memory before loading
    memory_before = measure_memory()
    start_time = time.time()
    
    # Initialize inference with timeout
    print("Initializing BinderInference...")
    try:
        inf = run_with_timeout(lambda: BinderInference(config_path=config_path), 300)  # 5 min timeout
        print("✅ BinderInference initialized successfully")
    except TimeoutException as e:
        print(f"❌ BinderInference initialization timed out: {e}")
        return None
    except Exception as e:
        print(f"❌ BinderInference initialization failed: {e}")
        return None
    
    init_time = time.time() - start_time
    memory_after_init = measure_memory()
    
    print(f"Initialization time: {init_time:.2f}s")
    print(f"Memory usage - Before: {memory_before:.1f}MB, After init: {memory_after_init:.1f}MB")
    
    # Warmup with timeout
    if warmup:
        print("Warming up with timeout protection...")
        try:
            def warmup_func():
                if hasattr(inf, 'warm_up'):
                    inf.warm_up()
                else:
                    inf.predict("Test warmup text")
                return True
            
            run_with_timeout(warmup_func, 60)  # 1 minute timeout for warmup
            print("✅ Warmup completed successfully")
        except TimeoutException as e:
            print(f"⚠️  Warmup timed out: {e}. Continuing without warmup.")
        except Exception as e:
            print(f"⚠️  Warmup failed: {e}. Continuing without warmup.")
    
    # Benchmark single predictions with individual timeouts
    print("Starting single prediction benchmarks...")
    single_times = []
    failed_predictions = 0
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        times_per_run = []
        for i, text in enumerate(texts):
            print(f"  Processing text {i + 1}/{len(texts)} (length: {len(text)})")
            try:
                def predict_single():
                    start = time.time()
                    predictions = inf.predict(text)
                    end = time.time()
                    return end - start, predictions
                
                pred_time, predictions = run_with_timeout(predict_single, 30)  # 30 sec per prediction
                times_per_run.append(pred_time)
                print(f"    ✅ Completed in {pred_time:.3f}s, found {len(predictions) if predictions else 0} entities")
                
            except TimeoutException as e:
                print(f"    ⚠️  Prediction timed out: {e}")
                failed_predictions += 1
            except Exception as e:
                print(f"    ⚠️  Prediction failed: {e}")
                failed_predictions += 1
        
        single_times.extend(times_per_run)
        print(f"  Run {run + 1} completed: {len(times_per_run)}/{len(texts)} successful predictions")
    
    # Benchmark batch predictions (if available) with timeout
    batch_times = []
    if hasattr(inf, 'predict_batch'):
        print("Starting batch prediction benchmarks...")
        for run in range(num_runs):
            try:
                def predict_batch():
                    start = time.time()
                    batch_predictions = inf.predict_batch(texts)
                    end = time.time()
                    return end - start, batch_predictions
                
                batch_time, batch_predictions = run_with_timeout(predict_batch, 60)  # 1 min for batch
                batch_times.append(batch_time)
                print(f"  ✅ Batch run {run + 1} completed in {batch_time:.3f}s")
                
            except TimeoutException as e:
                print(f"  ⚠️  Batch prediction timed out: {e}")
            except Exception as e:
                print(f"  ⚠️  Batch prediction failed: {e}")
    
    memory_peak = measure_memory()
    
    # Results
    if single_times:
        avg_single = statistics.mean(single_times)
        std_single = statistics.stdev(single_times) if len(single_times) > 1 else 0
        
        print(f"\nSingle prediction performance:")
        print(f"  Average time: {avg_single:.3f}s ± {std_single:.3f}s")
        print(f"  Successful predictions: {len(single_times)}/{len(texts) * num_runs}")
        print(f"  Failed predictions: {failed_predictions}")
        if single_times:
            print(f"  Total time: {sum(single_times):.2f}s")
            print(f"  Throughput: {len(single_times) / sum(single_times):.2f} predictions/sec")
    else:
        print(f"\n❌ No successful single predictions completed")
        avg_single = 0
    
    if batch_times:
        avg_batch = statistics.mean(batch_times)
        std_batch = statistics.stdev(batch_times) if len(batch_times) > 1 else 0
        print(f"\nBatch prediction performance:")
        print(f"  Average time: {avg_batch:.3f}s ± {std_batch:.3f}s")
        print(f"  Successful batches: {len(batch_times)}/{num_runs}")
        if batch_times:
            print(f"  Throughput: {len(texts) * len(batch_times) / sum(batch_times):.2f} texts/sec")
            if single_times:
                print(f"  Speedup: {sum(single_times) / sum(batch_times):.2f}x")
    
    print(f"\nMemory usage:")
    print(f"  Peak: {memory_peak:.1f}MB")
    print(f"  Increase: {memory_peak - memory_before:.1f}MB")
    
    # Clear cache if available
    try:
        if hasattr(inf, 'clear_cache'):
            inf.clear_cache()
    except Exception as e:
        print(f"⚠️  Failed to clear cache: {e}")
    
    return {
        'init_time': init_time,
        'avg_single_time': avg_single,
        'total_single_time': sum(single_times) if single_times else 0,
        'avg_batch_time': avg_batch if batch_times else None,
        'total_batch_time': sum(batch_times) if batch_times else None,
        'memory_peak': memory_peak,
        'memory_increase': memory_peak - memory_before,
        'single_throughput': len(single_times) / sum(single_times) if single_times else 0,
        'batch_throughput': len(texts) * len(batch_times) / sum(batch_times) if batch_times else None,
        'failed_predictions': failed_predictions,
        'successful_predictions': len(single_times)
    }

def main():
    # Test texts of varying lengths
    test_texts = [
        "Президент России Владимир Путин встретился с министром.",
        "В результате кибератаки была нарушена работа критически важной информационной инфраструктуры. Хакерская группа использовала неизвестную ранее уязвимость для проникновения в защищенные системы организации.",
        "Специалисты по информационной безопасности обнаружили новую программу-вымогатель, которая распространяется через фишинговые электронные письма. Malware шифрует файлы пользователей и требует выкуп в криптовалюте для их восстановления.",
        "Исследователи безопасности из компании опубликовали отчет о серии целенаправленных атак на финансовые учреждения. Атакующие использовали социальную инженерию и продвинутые методы обхода систем защиты.",
        "Центр мониторинга и реагирования на компьютерные атаки зафиксировал увеличение количества инцидентов, связанных с компрометацией учетных записей администраторов в корпоративных сетях российских организаций."
    ]
    
    print("=== BinderInference Performance Benchmark ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Number of test texts: {len(test_texts)}")
    print(f"CPU cores: {psutil.cpu_count()}")
    
    # Test with just the original config first
    print("\n" + "="*60)
    print("TESTING SINGLE CONFIG")
    print("="*60)
    
    original_results = benchmark_inference(
        "./inference/inference-config.json", 
        test_texts, 
        num_runs=2  # Reduced runs for faster testing
    )
    
    if original_results is None:
        print("❌ Original config benchmark failed completely")
        return
    
    # Only try optimized if original worked
    if original_results['successful_predictions'] > 0:
        print("\n" + "="*60)
        print("TESTING OPTIMIZED CONFIG")
        print("="*60)
        
        try:
            optimized_results = benchmark_inference(
                "./inference/inference-config-optimized.json", 
                test_texts, 
                num_runs=2
            )
            
            if optimized_results and original_results['successful_predictions'] > 0:
                # Compare results
                print("\n" + "="*60)
                print("PERFORMANCE COMPARISON")
                print("="*60)
                
                init_speedup = original_results['init_time'] / optimized_results['init_time'] if optimized_results['init_time'] > 0 else 1
                single_speedup = original_results['avg_single_time'] / optimized_results['avg_single_time'] if optimized_results['avg_single_time'] > 0 else 1
                throughput_improvement = optimized_results['single_throughput'] / original_results['single_throughput'] if original_results['single_throughput'] > 0 else 1
                memory_improvement = (original_results['memory_increase'] - optimized_results['memory_increase']) / original_results['memory_increase'] * 100 if original_results['memory_increase'] > 0 else 0
                
                print(f"Initialization speedup: {init_speedup:.2f}x")
                print(f"Single prediction speedup: {single_speedup:.2f}x")
                print(f"Throughput improvement: {throughput_improvement:.2f}x")
                print(f"Memory reduction: {memory_improvement:.1f}%")
                
                if optimized_results['batch_throughput']:
                    batch_vs_single = optimized_results['batch_throughput'] / optimized_results['single_throughput']
                    print(f"Batch vs single (optimized): {batch_vs_single:.2f}x")
        
        except Exception as e:
            print(f"⚠️  Optimized config test failed: {e}")
    
    print("\nOptimizations applied:")
    print("✓ Lazy loading of models and tokenizers")
    print("✓ Feature caching for repeated texts")
    print("✓ Batch processing capabilities")
    print("✓ Memory-efficient padding")
    print("✓ GPU optimizations (if available)")
    print("✓ Reduced redundant computations")
    print("✓ Timeout protection for all operations")
    print("✓ Graceful error handling")

if __name__ == "__main__":
    main() 