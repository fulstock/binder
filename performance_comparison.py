#!/usr/bin/env python3
"""
Performance comparison script for BinderInference implementations.
Compares text2ner.py (optimized) vs binder_proc.py (original) implementations.
"""

import os
import sys
import time
import psutil
import gc
import statistics
from typing import Dict, List, Tuple
import importlib.util

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def import_module_from_file(file_path: str, module_name: str):
    """Import a module from a file path"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Add to sys.modules to avoid reimport issues
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing {module_name} from {file_path}: {e}")
        raise

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m {seconds%60:.1f}s"
    else:
        return f"{seconds/3600:.1f}h {(seconds%3600)/60:.1f}m"

def collect_test_files(dataset_path: str, max_files: int = 50) -> List[str]:
    """Collect text files for testing"""
    text_files = []
    
    # Try both relative paths from the two scripts
    possible_paths = [
        dataset_path,
        os.path.join(".", dataset_path),
        os.path.join("..", dataset_path.replace("../", "")),
        "./data/seccol_events_texts_1500_new2-div/test",
        "../data/seccol/seccol_events_texts_1500_new2-div/test"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found dataset at: {path}")
            for root, dirs, files in os.walk(path):
                for f in sorted(files):
                    if f.endswith(".txt") and len(text_files) < max_files:
                        text_files.append(os.path.join(root, f))
                    if len(text_files) >= max_files:
                        break
                if len(text_files) >= max_files:
                    break
            break
    
    return text_files

def test_inference_performance(inference_class, config_path: str, test_files: List[str], 
                              name: str, warmup: bool = True) -> Dict:
    """Test inference performance for a given implementation"""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    results = {
        'name': name,
        'config_path': config_path,
        'init_time': 0,
        'init_memory': 0,
        'post_init_memory': 0,
        'warmup_time': 0,
        'processing_time': 0,
        'total_files': len(test_files),
        'processed_files': 0,
        'failed_files': 0,
        'total_chars': 0,
        'total_entities': 0,
        'files_per_second': 0,
        'chars_per_second': 0,
        'entities_per_file': 0,
        'peak_memory': 0,
        'error': None
    }
    
    # Check if config file exists
    if not os.path.exists(config_path):
        results['error'] = f"Config file not found: {config_path}"
        return results
    
    try:
        # Measure initialization
        print(f"Initializing {name}...")
        gc.collect()
        init_memory = get_memory_usage()
        results['init_memory'] = init_memory
        
        init_start = time.time()
        inf = inference_class(config_path=config_path, device="cpu")  # Use CPU for fair comparison
        init_time = time.time() - init_start
        
        results['init_time'] = init_time
        results['post_init_memory'] = get_memory_usage()
        
        print(f"  • Initialization: {format_time(init_time)}")
        print(f"  • Memory usage: {results['init_memory']:.1f}MB → {results['post_init_memory']:.1f}MB")
        
        # Warmup if requested
        if warmup:
            print("  • Warming up...")
            warmup_start = time.time()
            try:
                _ = inf.predict("Это тестовый текст для прогрева модели.")
                results['warmup_time'] = time.time() - warmup_start
                print(f"  • Warmup: {format_time(results['warmup_time'])}")
            except Exception as e:
                print(f"  • Warmup failed: {e}")
        
        # Process test files
        print(f"Processing {len(test_files)} files...")
        processing_start = time.time()
        
        for i, file_path in enumerate(test_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if not text:
                    continue
                
                # Skip very large files to prevent memory issues
                if len(text) > 100000:  # 100KB limit
                    continue
                
                # Predict entities
                predictions = inf.predict(text)
                
                # Update statistics
                results['processed_files'] += 1
                results['total_chars'] += len(text)
                results['total_entities'] += len(predictions)
                
                # Monitor memory usage
                current_memory = get_memory_usage()
                results['peak_memory'] = max(results['peak_memory'], current_memory)
                
                # Print progress every 10 files
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - processing_start
                    progress = (i + 1) / len(test_files) * 100
                    print(f"  • Progress: {progress:.1f}% ({i+1}/{len(test_files)}) - {elapsed:.1f}s elapsed")
                
            except Exception as e:
                results['failed_files'] += 1
                print(f"  • Failed to process {os.path.basename(file_path)}: {e}")
        
        processing_time = time.time() - processing_start
        results['processing_time'] = processing_time
        
        # Calculate throughput metrics
        if results['processed_files'] > 0:
            results['files_per_second'] = results['processed_files'] / processing_time
            results['chars_per_second'] = results['total_chars'] / processing_time
            results['entities_per_file'] = results['total_entities'] / results['processed_files']
        
        print(f"\n{name} Results:")
        print(f"  • Files processed: {results['processed_files']}/{results['total_files']}")
        print(f"  • Files failed: {results['failed_files']}")
        print(f"  • Processing time: {format_time(processing_time)}")
        print(f"  • Throughput: {results['files_per_second']:.2f} files/s")
        print(f"  • Character throughput: {results['chars_per_second']:.0f} chars/s")
        print(f"  • Entities found: {results['total_entities']} ({results['entities_per_file']:.1f} per file)")
        print(f"  • Peak memory: {results['peak_memory']:.1f}MB")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"  • Error: {e}")
    
    return results

def compare_results(results1: Dict, results2: Dict):
    """Compare results between two implementations"""
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    # Basic comparison
    print(f"{'Metric':<25} {'Text2NER (Optimized)':<20} {'Binder_Proc (Original)':<20} {'Speedup':<10}")
    print("-" * 80)
    
    # Initialization time
    if results1['init_time'] > 0 and results2['init_time'] > 0:
        speedup = results2['init_time'] / results1['init_time']
        print(f"{'Init Time':<25} {format_time(results1['init_time']):<20} {format_time(results2['init_time']):<20} {speedup:.2f}x")
    
    # Processing time
    if results1['processing_time'] > 0 and results2['processing_time'] > 0:
        speedup = results2['processing_time'] / results1['processing_time']
        print(f"{'Processing Time':<25} {format_time(results1['processing_time']):<20} {format_time(results2['processing_time']):<20} {speedup:.2f}x")
    
    # Files per second
    if results1['files_per_second'] > 0 and results2['files_per_second'] > 0:
        speedup = results1['files_per_second'] / results2['files_per_second']
        val1 = f"{results1['files_per_second']:.2f}"
        val2 = f"{results2['files_per_second']:.2f}"
        print(f"{'Files/Second':<25} {val1:<20} {val2:<20} {speedup:.2f}x")
    
    # Characters per second
    if results1['chars_per_second'] > 0 and results2['chars_per_second'] > 0:
        speedup = results1['chars_per_second'] / results2['chars_per_second']
        val1 = f"{results1['chars_per_second']:.0f}"
        val2 = f"{results2['chars_per_second']:.0f}"
        print(f"{'Chars/Second':<25} {val1:<20} {val2:<20} {speedup:.2f}x")
    
    # Memory usage
    val1 = f"{results1['init_memory']:.1f}"
    val2 = f"{results2['init_memory']:.1f}"
    print(f"{'Init Memory (MB)':<25} {val1:<20} {val2:<20} {'-':<10}")
    
    val1 = f"{results1['post_init_memory']:.1f}"
    val2 = f"{results2['post_init_memory']:.1f}"
    print(f"{'Post-Init Memory (MB)':<25} {val1:<20} {val2:<20} {'-':<10}")
    
    val1 = f"{results1['peak_memory']:.1f}"
    val2 = f"{results2['peak_memory']:.1f}"
    print(f"{'Peak Memory (MB)':<25} {val1:<20} {val2:<20} {'-':<10}")
    
    # Success rates
    success_rate1 = (results1['processed_files'] / results1['total_files']) * 100 if results1['total_files'] > 0 else 0
    success_rate2 = (results2['processed_files'] / results2['total_files']) * 100 if results2['total_files'] > 0 else 0
    val1 = f"{success_rate1:.1f}"
    val2 = f"{success_rate2:.1f}"
    print(f"{'Success Rate (%)':<25} {val1:<20} {val2:<20} {'-':<10}")
    
    # Quality metrics
    val1 = f"{results1['entities_per_file']:.1f}"
    val2 = f"{results2['entities_per_file']:.1f}"
    print(f"{'Entities per File':<25} {val1:<20} {val2:<20} {'-':<10}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if results1['error'] and results2['error']:
        print("❌ Both implementations failed")
    elif results1['error']:
        print("❌ Text2NER (Optimized) failed")
        print(f"✅ Binder_Proc (Original) completed successfully")
    elif results2['error']:
        print("❌ Binder_Proc (Original) failed")
        print("✅ Text2NER (Optimized) completed successfully")
    else:
        print("✅ Both implementations completed successfully")
        
        # Determine winner
        if results1['files_per_second'] > results2['files_per_second']:
            winner = "Text2NER (Optimized)"
            speedup = results1['files_per_second'] / results2['files_per_second']
        else:
            winner = "Binder_Proc (Original)"
            speedup = results2['files_per_second'] / results1['files_per_second']
        
        print(f"🏆 Winner: {winner} ({speedup:.2f}x faster)")

def main():
    """Main function to run the performance comparison"""
    print("BinderInference Performance Comparison")
    print("Comparing text2ner.py vs binder_proc.py implementations")
    
    # Import both modules
    print("\nImporting modules...")
    try:
        text2ner_module = import_module_from_file("text2ner.py", "text2ner")
        binder_proc_module = import_module_from_file("binder_proc.py", "binder_proc")
        print("✅ Successfully imported both modules")
    except Exception as e:
        print(f"❌ Failed to import modules: {e}")
        return
    
    # Get test files
    print("\nCollecting test files...")
    test_files = collect_test_files("./data/seccol_events_texts_1500_new2-div/test", max_files=20)
    
    if not test_files:
        print("❌ No test files found. Please check the dataset paths.")
        return
    
    print(f"✅ Found {len(test_files)} test files")
    
    # Test both implementations
    print(f"\nStarting performance comparison...")
    
    # Test text2ner.py (optimized)
    results1 = test_inference_performance(
        text2ner_module.BinderInference,
        "./inference/inference-config-optimized.json",
        test_files,
        "Text2NER (Optimized)",
        warmup=True
    )
    
    # Force garbage collection between tests
    gc.collect()
    time.sleep(2)  # Allow system to stabilize
    
    # Test binder_proc.py (original)
    results2 = test_inference_performance(
        binder_proc_module.BinderInference,
        "./inference/inference-config.json",
        test_files,
        "Binder_Proc (Original)",
        warmup=True
    )
    
    # Compare results
    compare_results(results1, results2)
    
    # Final system stats
    print(f"\n{'='*80}")
    print("SYSTEM INFORMATION")
    print(f"{'='*80}")
    
    # System memory info
    mem = psutil.virtual_memory()
    print(f"System Memory: {mem.total / 1024**3:.1f}GB total, {mem.available / 1024**3:.1f}GB available")
    
    # CPU info
    print(f"CPU Count: {psutil.cpu_count()} cores")
    
    print(f"\n✅ Performance comparison completed!")

if __name__ == "__main__":
    main() 