"""
Performance Comparison: AI-Generated vs Manual Implementation
"""

import timeit
import sys
import tracemalloc
import ast
import os

# Import both implementations
from ai_generated import sort_dictionaries_by_key as ai_sort
from manual_implementation import sort_dictionaries_by_key as manual_sort

def generate_test_data(size: int = 1000):
    """Generate test data for performance comparison."""
    import random
    
    data = []
    for i in range(size):
        data.append({
            'id': i,
            'name': f'User_{random.randint(1, 1000)}',
            'age': random.randint(18, 80),
            'score': round(random.uniform(0, 100), 2),
            'category': random.choice(['A', 'B', 'C', 'D'])
        })
    return data

def measure_execution_time():
    """Measure execution time for both implementations."""
    test_data = generate_test_data(1000)
    
    # AI implementation timing
    ai_time = timeit.timeit(
        lambda: ai_sort(test_data.copy(), 'score'),
        number=100
    )
    
    # Manual implementation timing  
    manual_time = timeit.timeit(
        lambda: manual_sort(test_data.copy(), 'score'),
        number=100
    )
    
    return ai_time, manual_time

def measure_memory_usage():
    """Measure memory usage for both implementations."""
    test_data = generate_test_data(5000)
    
    # AI implementation memory
    tracemalloc.start()
    ai_sort(test_data.copy(), 'score')
    ai_current, ai_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Manual implementation memory
    tracemalloc.start()
    manual_sort(test_data.copy(), 'score')
    manual_current, manual_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return (ai_current, ai_peak), (manual_current, manual_peak)

def calculate_complexity():
    """Calculate cyclomatic complexity for both implementations."""
    
    def count_complexity(code):
        """Simple complexity calculation based on control structures."""
        complexity = 1  # Base complexity
        
        # Count decision points
        complexity += code.count('if ')
        complexity += code.count('elif ')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('except ')
        complexity += code.count('and ')
        complexity += code.count('or ')
        
        return complexity
    
    # Read AI implementation
    with open('ai_generated.py', 'r') as f:
        ai_code = f.read()
    
    # Read manual implementation
    with open('manual_implementation.py', 'r') as f:
        manual_code = f.read()
    
    # Calculate complexity for main functions only
    ai_func_start = ai_code.find('def sort_dictionaries_by_key')
    ai_func_end = ai_code.find('def test_ai_generated', ai_func_start)
    ai_func_code = ai_code[ai_func_start:ai_func_end] if ai_func_end > 0 else ai_code[ai_func_start:]
    
    manual_func_start = manual_code.find('def sort_dictionaries_by_key')
    manual_func_end = manual_code.find('def test_manual_implementation', manual_func_start)
    manual_func_code = manual_code[manual_func_start:manual_func_end] if manual_func_end > 0 else manual_code[manual_func_start:]
    
    ai_complexity = count_complexity(ai_func_code)
    manual_complexity = count_complexity(manual_func_code)
    
    return ai_complexity, manual_complexity

def count_lines_of_code():
    """Count lines of code for both implementations."""
    
    def count_loc(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Count non-empty, non-comment lines
        code_lines = 0
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            elif stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                comment_lines += 1
            else:
                code_lines += 1
        
        return code_lines, comment_lines
    
    ai_code, ai_comments = count_loc('ai_generated.py')
    manual_code, manual_comments = count_loc('manual_implementation.py')
    
    return (ai_code, ai_comments), (manual_code, manual_comments)

def run_comparison():
    """Run complete performance comparison."""
    print("=" * 60)
    print("PERFORMANCE COMPARISON: AI-Generated vs Manual Implementation")
    print("=" * 60)
    
    # Execution time comparison
    print("\n1. EXECUTION TIME COMPARISON")
    print("-" * 30)
    ai_time, manual_time = measure_execution_time()
    print(f"AI Implementation:     {ai_time:.6f} seconds (100 runs)")
    print(f"Manual Implementation: {manual_time:.6f} seconds (100 runs)")
    print(f"Performance Ratio:     {ai_time/manual_time:.2f}x")
    
    # Memory usage comparison
    print("\n2. MEMORY USAGE COMPARISON")
    print("-" * 30)
    (ai_current, ai_peak), (manual_current, manual_peak) = measure_memory_usage()
    print(f"AI Implementation:")
    print(f"  Current: {ai_current/1024:.2f} KB, Peak: {ai_peak/1024:.2f} KB")
    print(f"Manual Implementation:")
    print(f"  Current: {manual_current/1024:.2f} KB, Peak: {manual_peak/1024:.2f} KB")
    
    # Complexity comparison
    print("\n3. CYCLOMATIC COMPLEXITY")
    print("-" * 30)
    ai_complexity, manual_complexity = calculate_complexity()
    print(f"AI Implementation:     {ai_complexity}")
    print(f"Manual Implementation: {manual_complexity}")
    
    # Lines of code comparison
    print("\n4. LINES OF CODE")
    print("-" * 30)
    (ai_code, ai_comments), (manual_code, manual_comments) = count_lines_of_code()
    print(f"AI Implementation:")
    print(f"  Code: {ai_code} lines, Comments: {ai_comments} lines")
    print(f"  Comment ratio: {(ai_comments/(ai_code+ai_comments)*100):.1f}%")
    print(f"Manual Implementation:")
    print(f"  Code: {manual_code} lines, Comments: {manual_comments} lines")
    print(f"  Comment ratio: {(manual_comments/(manual_code+manual_comments)*100):.1f}%")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_comparison()