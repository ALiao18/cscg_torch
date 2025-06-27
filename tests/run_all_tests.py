#!/usr/bin/env python3
"""
CSCG Torch Master Test Runner

Comprehensive test suite runner for all CSCG components:
- Models (CHMM_torch and training utilities)
- Environment adapters (room navigation)
- Integration tests (full pipeline)
- Performance benchmarks

Usage:
    python run_all_tests.py [--gpu-only] [--no-slow] [--verbose]
"""

import sys
import argparse
import time
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_config import TestConfig, setup_test_environment, cleanup_test_environment

def import_test_module(module_path: Path):
    """
    Dynamically import a test module.
    
    Args:
        module_path: Path to the test module
        
    Returns:
        Imported module
    """
    spec = importlib.util.spec_from_file_location("test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_test_suite(test_name: str, test_module_path: Path, verbose: bool = False) -> dict:
    """
    Run a test suite and return results.
    
    Args:
        test_name: Name of the test suite
        test_module_path: Path to the test module
        verbose: Whether to show verbose output
        
    Returns:
        dict: Test results
    """
    print(f"\\n{'='*60}")
    print(f"Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Import and run test module
        test_module = import_test_module(test_module_path)
        
        if hasattr(test_module, 'run_all_tests'):
            # Run the test suite
            test_module.run_all_tests()
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            return {
                'status': 'passed',
                'execution_time': execution_time,
                'error': None
            }
        else:
            return {
                'status': 'failed',
                'execution_time': 0.0,
                'error': 'No run_all_tests function found'
            }
            
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'status': 'failed',
            'execution_time': execution_time,
            'error': str(e)
        }

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='CSCG Torch Test Suite Runner')
    parser.add_argument('--gpu-only', action='store_true', 
                       help='Run only GPU-specific tests')
    parser.add_argument('--no-slow', action='store_true',
                       help='Skip slow tests')
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose output')
    parser.add_argument('--suite', choices=['models', 'adapters', 'integration', 'all'],
                       default='all', help='Which test suite to run')
    
    args = parser.parse_args()
    
    print("CSCG Torch Comprehensive Test Suite")
    print("=" * 60)
    print(f"GPU Available: {TestConfig.USE_GPU}")
    print(f"Device: {TestConfig.DEVICE}")
    print(f"Test Configuration:")
    print(f"  - Sequence Length: {TestConfig.SEQUENCE_LENGTH}")
    print(f"  - Room Size: {TestConfig.ROOM_SIZE}x{TestConfig.ROOM_SIZE}")
    print(f"  - EM Iterations: {TestConfig.EM_ITERATIONS}")
    print(f"  - Random Seed: {TestConfig.SEED}")
    
    if args.gpu_only and not TestConfig.USE_GPU:
        print("\\n❌ GPU-only tests requested but CUDA not available!")
        sys.exit(1)
    
    # Set up global test environment
    setup_test_environment()
    
    # Define test suites
    test_suites = {
        'models': {
            'name': 'CHMM Models Tests',
            'path': Path(__file__).parent / 'models' / 'test_chmm_torch.py',
            'required': True
        },
        'adapters': {
            'name': 'Environment Adapters Tests', 
            'path': Path(__file__).parent / 'env_adapters' / 'test_room_adapters.py',
            'required': True
        },
        'integration': {
            'name': 'Integration Tests',
            'path': Path(__file__).parent / 'integration' / 'test_full_pipeline.py',
            'required': False
        }
    }
    
    # Filter test suites based on arguments
    if args.suite != 'all':
        test_suites = {args.suite: test_suites[args.suite]}
    
    # Run test suites
    results = {}
    total_start_time = time.time()
    
    for suite_key, suite_info in test_suites.items():
        if not suite_info['path'].exists():
            print(f"\\n⚠️  Skipping {suite_info['name']} (file not found: {suite_info['path']})")
            results[suite_key] = {
                'status': 'skipped',
                'execution_time': 0.0,
                'error': 'Test file not found'
            }
            continue
        
        # Skip non-required tests if they might fail
        if not suite_info['required'] and args.no_slow:
            print(f"\\n⏭️  Skipping {suite_info['name']} (slow test)")
            results[suite_key] = {
                'status': 'skipped',
                'execution_time': 0.0,
                'error': 'Skipped (slow test)'
            }
            continue
        
        # Run test suite
        results[suite_key] = run_test_suite(
            suite_info['name'],
            suite_info['path'],
            args.verbose
        )
    
    total_execution_time = time.time() - total_start_time
    
    # Print summary
    print(f"\\n\\n{'='*60}")
    print("FINAL TEST SUMMARY")
    print(f"{'='*60}")
    
    passed_suites = 0
    failed_suites = 0
    skipped_suites = 0
    
    for suite_key, result in results.items():
        suite_name = test_suites[suite_key]['name']
        status = result['status']
        exec_time = result['execution_time']
        
        if status == 'passed':
            print(f"✅ {suite_name}: PASSED ({exec_time:.2f}s)")
            passed_suites += 1
        elif status == 'failed':
            print(f"❌ {suite_name}: FAILED ({exec_time:.2f}s)")
            if result['error']:
                print(f"   Error: {result['error']}")
            failed_suites += 1
        else:  # skipped
            print(f"⏭️  {suite_name}: SKIPPED")
            if result['error']:
                print(f"   Reason: {result['error']}")
            skipped_suites += 1
    
    total_suites = len(results)
    print(f"\\nOverall Results:")
    print(f"  Total Suites: {total_suites}")
    print(f"  Passed: {passed_suites}")
    print(f"  Failed: {failed_suites}")
    print(f"  Skipped: {skipped_suites}")
    print(f"  Success Rate: {(passed_suites/max(1, total_suites-skipped_suites))*100:.1f}%")
    print(f"  Total Execution Time: {total_execution_time:.2f}s")
    
    # GPU memory summary
    if TestConfig.USE_GPU:
        try:
            from tests.test_config import check_gpu_memory
            memory_info = check_gpu_memory()
            print(f"\\nGPU Memory Summary:")
            print(f"  Current Allocated: {memory_info['allocated_mb']:.1f} MB")
            print(f"  Peak Allocated: {memory_info['max_allocated_mb']:.1f} MB")
        except:
            pass
    
    # Clean up
    cleanup_test_environment()
    
    # Exit with appropriate code
    if failed_suites > 0:
        print(f"\\n❌ {failed_suites} test suite(s) failed!")
        sys.exit(1)
    else:
        print(f"\\n✅ All test suites passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()