"""Test script to check if the father name mapping works correctly."""

import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

# Path to the father name transliteration model
model_path = Path(__file__).resolve().parent / "models" / "father_name_transliteration_model_corrected.pkl"

def load_and_test_model():
    """Load the father name model and test the specific mapping."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"Loaded father name transliteration model with {len(model['direct_mappings'])} mappings")
            
            test_name = "শ্রী অম্বী চন্দ্র সরকার"
            print(f"Checking mapping for: '{test_name}'")
            
            # Check in direct mappings
            if test_name in model['direct_mappings']:
                print(f"Direct mapping: {test_name} -> {model['direct_mappings'][test_name]}")
            else:
                print(f"No direct mapping found for: '{test_name}'")
            
            # Check individual words
            words = test_name.split()
            for word in words:
                if word in model['direct_mappings']:
                    print(f"Word mapping: {word} -> {model['direct_mappings'][word]}")
                else:
                    print(f"No mapping for word: '{word}'")
            
            # Also check in NAME_MAPPINGS from the translator file
            from TransBnEn.translateIndigo_with_father_name import NAME_MAPPINGS
            print("\nChecking in NAME_MAPPINGS dictionary:")
            if test_name in NAME_MAPPINGS:
                print(f"Direct mapping in NAME_MAPPINGS: {test_name} -> {NAME_MAPPINGS[test_name]}")
            else:
                print(f"No direct mapping in NAME_MAPPINGS for: '{test_name}'")
            
            for word in words:
                if word in NAME_MAPPINGS:
                    print(f"Word mapping in NAME_MAPPINGS: {word} -> {NAME_MAPPINGS[word]}")
                else:
                    print(f"No mapping in NAME_MAPPINGS for word: '{word}'")
                    
    except Exception as e:
        print(f"Error loading or testing model: {e}")

if __name__ == "__main__":
    load_and_test_model()
