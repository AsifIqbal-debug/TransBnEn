"""Script to test the mapping for 'মোছাঃ' in our translator."""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent))

# Import the translator function directly
from TransBnEn.translateIndigo_with_name import NAME_MAPPINGS, transliterate_name_specialized

def test_mapping():
    """Test the mapping for 'মোছাঃ'."""
    test_word = "মোছাঃ"
    print(f"\nTesting translation for: '{test_word}'")
    
    # Check if it's in NAME_MAPPINGS
    if test_word in NAME_MAPPINGS:
        print(f"Found in NAME_MAPPINGS: '{test_word}' -> '{NAME_MAPPINGS[test_word]}'")
    else:
        print(f"Not found in NAME_MAPPINGS!")
    
    # Test with the specialized name transliteration function
    result = transliterate_name_specialized(test_word)
    print(f"Using transliterate_name_specialized(): '{test_word}' -> '{result}'")
    
    # Test with a full name
    full_name = "মোছাঃ তহমিনা বেগম"
    print(f"\nTesting full name: '{full_name}'")
    result = transliterate_name_specialized(full_name)
    print(f"Using transliterate_name_specialized(): '{full_name}' -> '{result}'")

if __name__ == "__main__":
    test_mapping()
