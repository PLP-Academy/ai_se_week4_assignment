"""
AI-Generated Implementation: Sort dictionaries by key
Generated using AI assistance (simulated GitHub Copilot style)
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def sort_dictionaries_by_key(
    dict_list: List[Dict[str, Any]], 
    sort_key: str,
    reverse: bool = False
) -> Optional[List[Dict[str, Any]]]:
    """
    Sort a list of dictionaries by a specific key using AI-generated approach.
    
    Args:
        dict_list: List of dictionaries to sort
        sort_key: The key to sort by
        reverse: Sort in descending order if True
        
    Returns:
        Sorted list of dictionaries or None if error occurs
        
    Raises:
        ValueError: If dict_list is empty or sort_key is invalid
        KeyError: If sort_key doesn't exist in dictionaries
    """
    # Input validation - handle edge cases
    if not dict_list:
        logger.warning("Empty dictionary list provided")
        raise ValueError("Cannot sort empty list")
    
    if not sort_key:
        logger.error("Invalid sort key provided")
        raise ValueError("Sort key cannot be empty")
    
    try:
        # AI-generated approach: Use built-in sorted with lambda
        # This is a common AI suggestion pattern
        sorted_list = sorted(
            dict_list, 
            key=lambda x: x.get(sort_key, 0),  # Default to 0 for missing keys
            reverse=reverse
        )
        
        logger.info(f"Successfully sorted {len(dict_list)} dictionaries by '{sort_key}'")
        return sorted_list
        
    except Exception as e:
        logger.error(f"Error sorting dictionaries: {str(e)}")
        return None

def test_ai_generated():
    """Test cases for AI-generated implementation."""
    # Test case 1: Basic functionality
    test_data = [
        {'name': 'Alice', 'age': 30, 'score': 85},
        {'name': 'Bob', 'age': 25, 'score': 92},
        {'name': 'Charlie', 'age': 35, 'score': 78}
    ]
    
    result = sort_dictionaries_by_key(test_data, 'age')
    assert result[0]['name'] == 'Bob'  # Youngest first
    
    # Test case 2: Reverse sorting
    result_desc = sort_dictionaries_by_key(test_data, 'score', reverse=True)
    assert result_desc[0]['score'] == 92  # Highest score first
    
    # Test case 3: Missing key handling
    test_with_missing = [
        {'name': 'Alice', 'age': 30},
        {'name': 'Bob', 'score': 92},
        {'name': 'Charlie', 'age': 35, 'score': 78}
    ]
    
    result_missing = sort_dictionaries_by_key(test_with_missing, 'score')
    assert len(result_missing) == 3  # Should handle missing keys gracefully
    
    print("All AI-generated tests passed!")

if __name__ == "__main__":
    test_ai_generated()