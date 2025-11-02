"""
Manual Implementation: Sort dictionaries by key
Optimized manual implementation with advanced error handling
"""

from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

def sort_dictionaries_by_key(
    dict_list: List[Dict[str, Any]], 
    sort_key: str,
    reverse: bool = False,
    handle_missing: str = 'error'
) -> List[Dict[str, Any]]:
    """
    Sort a list of dictionaries by a specific key with optimized manual implementation.
    
    Args:
        dict_list: List of dictionaries to sort
        sort_key: The key to sort by
        reverse: Sort in descending order if True
        handle_missing: How to handle missing keys ('error', 'skip', 'last', 'default')
        
    Returns:
        Sorted list of dictionaries
        
    Raises:
        ValueError: If inputs are invalid
        KeyError: If sort_key missing and handle_missing='error'
    """
    # Comprehensive input validation
    if not isinstance(dict_list, list):
        raise ValueError("dict_list must be a list")
    
    if not dict_list:
        logger.warning("Empty list provided")
        return []
    
    if not isinstance(sort_key, str) or not sort_key.strip():
        raise ValueError("sort_key must be a non-empty string")
    
    # Validate all items are dictionaries
    for i, item in enumerate(dict_list):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not a dictionary")
    
    # Handle missing keys strategy
    valid_items = []
    invalid_items = []
    
    for item in dict_list:
        if sort_key in item:
            valid_items.append(item)
        else:
            if handle_missing == 'error':
                raise KeyError(f"Key '{sort_key}' not found in dictionary: {item}")
            elif handle_missing == 'skip':
                continue  # Skip items without the key
            else:
                invalid_items.append(item)
    
    # Optimized sorting with type-aware comparison
    def get_sort_value(item: Dict[str, Any]) -> Union[int, float, str]:
        """Extract and normalize sort value for consistent comparison."""
        value = item[sort_key]
        
        # Handle None values
        if value is None:
            return float('-inf') if not reverse else float('inf')
        
        # Ensure consistent numeric comparison
        if isinstance(value, (int, float)):
            return value
        
        # Convert to string for consistent comparison
        return str(value).lower()
    
    # Sort valid items using optimized key function
    try:
        sorted_valid = sorted(valid_items, key=get_sort_value, reverse=reverse)
    except TypeError as e:
        logger.error(f"Cannot compare values for key '{sort_key}': {e}")
        raise ValueError(f"Incompatible data types for sorting key '{sort_key}'")
    
    # Handle invalid items based on strategy
    if handle_missing == 'last':
        result = sorted_valid + invalid_items
    else:
        result = sorted_valid
    
    logger.info(f"Sorted {len(result)} dictionaries by '{sort_key}' "
                f"({'descending' if reverse else 'ascending'})")
    
    return result

def test_manual_implementation():
    """Comprehensive test cases for manual implementation."""
    # Test case 1: Basic functionality with mixed data types
    test_data = [
        {'name': 'Alice', 'age': 30, 'score': 85.5},
        {'name': 'Bob', 'age': 25, 'score': 92.0},
        {'name': 'Charlie', 'age': 35, 'score': 78.2}
    ]
    
    result = sort_dictionaries_by_key(test_data, 'age')
    assert result[0]['name'] == 'Bob'
    assert len(result) == 3
    
    # Test case 2: String sorting (case-insensitive)
    result_name = sort_dictionaries_by_key(test_data, 'name')
    assert result_name[0]['name'] == 'Alice'
    
    # Test case 3: Handle missing keys with 'skip' strategy
    test_missing = [
        {'name': 'Alice', 'age': 30},
        {'name': 'Bob'},  # Missing age
        {'name': 'Charlie', 'age': 35}
    ]
    
    result_skip = sort_dictionaries_by_key(test_missing, 'age', handle_missing='skip')
    assert len(result_skip) == 2  # Bob should be skipped
    
    print("All manual implementation tests passed!")

if __name__ == "__main__":
    test_manual_implementation()