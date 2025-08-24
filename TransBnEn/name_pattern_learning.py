"""
Utility module to learn patterns from user feedback in Bengali-to-English transliteration.
"""

import re
import json
import os
from pathlib import Path

# Path to store learned patterns
LEARNED_PATTERNS_PATH = Path(__file__).resolve().parent.parent / "data" / "learned_patterns.json"

def ensure_data_dir():
    """Ensure the data directory exists"""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

def load_learned_patterns():
    """Load previously learned patterns"""
    try:
        if LEARNED_PATTERNS_PATH.exists():
            with open(LEARNED_PATTERNS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"patterns": [], "examples": {}}
    except Exception as e:
        print(f"Error loading learned patterns: {e}")
        return {"patterns": [], "examples": {}}

def save_learned_patterns(patterns_data):
    """Save learned patterns to file"""
    ensure_data_dir()
    try:
        with open(LEARNED_PATTERNS_PATH, 'w', encoding='utf-8') as f:
            json.dump(patterns_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving learned patterns: {e}")
        return False

def add_transliteration_example(bengali, english):
    """
    Add a new transliteration example to learn from.
    This will be used to generate patterns.
    """
    patterns_data = load_learned_patterns()
    
    # Add to examples
    patterns_data["examples"][bengali] = english
    
    # Try to generate patterns from examples
    _update_patterns_from_examples(patterns_data)
    
    # Save updated data
    return save_learned_patterns(patterns_data)

def _update_patterns_from_examples(patterns_data):
    """
    Update patterns based on examples.
    This is where the learning happens.
    """
    examples = patterns_data["examples"]
    
    # Look for common patterns in examples
    # 1. Fixed phrases
    for bn, en in examples.items():
        # Skip if already covered by a pattern
        if any(_match_pattern(p, bn) for p in patterns_data["patterns"]):
            continue
        
        # Add direct pattern for this example
        patterns_data["patterns"].append({
            "type": "direct",
            "bengali_pattern": re.escape(bn),
            "english_template": en,
            "examples": [bn]
        })
    
    # 2. Name prefix patterns (e.g., "মোঃ" followed by a name)
    _find_prefix_patterns(patterns_data)
    
    # 3. Name suffix patterns (e.g., name followed by "খাতুন")
    _find_suffix_patterns(patterns_data)

def _find_prefix_patterns(patterns_data):
    """Find common prefixes in the examples"""
    examples = patterns_data["examples"]
    prefixes = [
        "মোঃ", "শ্রী", "মোহাম্মদ", "মোছাঃ", "ড.", "প্রফে.", 
        "শ্রীমতী", "জনাব", "জনাবা", "কুমারী", "অধ্যাপক", 
        "সুশ্রী", "শ্রীযুক্ত", "ডাঃ", "প্রফেসর", "ডক্টর", "শেখ"
    ]
    
    for prefix in prefixes:
        prefix_examples = [(bn, en) for bn, en in examples.items() 
                          if bn.startswith(prefix + " ")]
        
        if prefix_examples:  # Even one example is valuable for prefixes
            # Try to find the corresponding English prefix
            en_prefixes = []
            for bn, en in prefix_examples:
                # Get the Bengali text after the prefix
                bn_remainder = bn[len(prefix) + 1:].strip()
                
                # Split both Bengali and English into words
                bn_words = bn_remainder.split()
                en_words = en.split()
                
                # If the number of words matches, assume prefix maps to first English word
                if len(bn_words) + 1 == len(en_words):  # +1 for the prefix
                    en_prefixes.append(en_words[0])
                # Special case for two-part prefixes like "Md."
                elif len(bn_words) + 1 < len(en_words) and en_words[0].endswith('.'):
                    en_prefixes.append(en_words[0])
            
            # If we found consistent English prefixes
            if en_prefixes and all(p == en_prefixes[0] for p in en_prefixes):
                en_prefix = en_prefixes[0]
                
                # Create a pattern
                patterns_data["patterns"].append({
                    "type": "prefix",
                    "bengali_pattern": f"^{re.escape(prefix)}\\s+(.*?)$",
                    "english_template": f"{en_prefix} {{1}}",
                    "examples": [bn for bn, _ in prefix_examples]
                })

def _find_suffix_patterns(patterns_data):
    """Find common suffixes in the examples"""
    examples = patterns_data["examples"]
    suffixes = [
        "খাতুন", "আরা", "ইসলাম", "রহমান", "উদ্দিন", "হোসেন", "বেগম",
        "আলী", "আহমেদ", "সরকার", "দাস", "আলম", "খান", "চৌধুরী", "বসু",
        "মজুমদার", "দত্ত", "মিত্র", "সেন", "পাল", "ঘোষ", "মণ্ডল", "রায়",
        "বিশ্বাস", "হালদার", "নাথ", "করিম", "কবীর", "আক্তার", "সুলতানা",
        "সরদার", "সাহা"
    ]
    
    for suffix in suffixes:
        suffix_examples = [(bn, en) for bn, en in examples.items() 
                          if bn.endswith(" " + suffix)]
        
        if suffix_examples:  # Even one example is valuable for suffixes
            # Try to find the corresponding English suffix
            en_suffixes = []
            for bn, en in suffix_examples:
                # Get the Bengali text before the suffix
                bn_prefix = bn[:-len(suffix) - 1].strip()
                
                # Split both Bengali and English into words
                bn_words = bn_prefix.split()
                en_words = en.split()
                
                # If the number of words matches, assume suffix maps to last English word
                if len(bn_words) + 1 == len(en_words):  # +1 for the suffix
                    en_suffixes.append(en_words[-1])
            
            # If we found consistent English suffixes
            if en_suffixes and all(s == en_suffixes[0] for s in en_suffixes):
                en_suffix = en_suffixes[0]
                
                # Create a pattern
                patterns_data["patterns"].append({
                    "type": "suffix",
                    "bengali_pattern": f"^(.*?)\\s+{re.escape(suffix)}$",
                    "english_template": f"{{1}} {en_suffix}",
                    "examples": [bn for bn, _ in suffix_examples]
                })

def _match_pattern(pattern, text):
    """Check if a pattern matches the given text"""
    try:
        return re.search(pattern["bengali_pattern"], text) is not None
    except:
        return False

def apply_learned_patterns(text):
    """
    Apply learned patterns to translate a Bengali text.
    Returns the transliterated text if a pattern matches, otherwise None.
    """
    patterns_data = load_learned_patterns()
    
    for pattern in patterns_data["patterns"]:
        match = re.search(pattern["bengali_pattern"], text)
        if match:
            if pattern["type"] == "direct":
                return pattern["english_template"]
            else:
                # Replace {1}, {2}, etc. with matched groups
                result = pattern["english_template"]
                for i, group in enumerate(match.groups(), 1):
                    result = result.replace(f"{{{i}}}", group)
                return result
                
    return None
