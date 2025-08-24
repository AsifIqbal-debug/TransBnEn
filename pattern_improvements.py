"""
Improvements for the pattern-based Bengali name transliteration system
based on testing results from 500 names.
"""

import re
from pathlib import Path
import json
from TransBnEn.translateIndigo_with_name import (
    transliterate, transliterate_name_specialized, 
    apply_name_patterns, init_name_patterns,
    NAME_PATTERNS
)

# Initialize the patterns
init_name_patterns()

def analyze_missed_pattern_opportunities(csv_path="transliteration_results_500.csv"):
    """Analyze names that weren't matched by patterns but could benefit from patterns."""
    import csv
    
    missed_opportunities = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['pattern'] == "No pattern match" and not row['methods_match']:
                # This name had no pattern match and inconsistent transliteration
                missed_opportunities.append({
                    'bengali': row['bengali_name'],
                    'simple': row['simple'],
                    'specialized': row['specialized'],
                    'general': row['general']
                })
    
    print(f"Found {len(missed_opportunities)} names that could benefit from pattern matching")
    
    # Group by common prefixes
    prefixes = {}
    for item in missed_opportunities:
        words = item['bengali'].split()
        if len(words) > 1:
            first_word = words[0]
            if first_word not in prefixes:
                prefixes[first_word] = []
            prefixes[first_word].append(item)
    
    # Find common prefixes that appear multiple times
    common_prefixes = {prefix: items for prefix, items in prefixes.items() if len(items) >= 2}
    
    print(f"\nCommon missed prefixes (appearing in 2+ names):")
    for prefix, items in sorted(common_prefixes.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  '{prefix}' appears in {len(items)} names")
    
    # Group by common suffixes
    suffixes = {}
    for item in missed_opportunities:
        words = item['bengali'].split()
        if len(words) > 1:
            last_word = words[-1]
            if last_word not in suffixes:
                suffixes[last_word] = []
            suffixes[last_word].append(item)
    
    # Find common suffixes that appear multiple times
    common_suffixes = {suffix: items for suffix, items in suffixes.items() if len(items) >= 2}
    
    print(f"\nCommon missed suffixes (appearing in 2+ names):")
    for suffix, items in sorted(common_suffixes.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  '{suffix}' appears in {len(items)} names")
    
    # Identify Ya-Phalaa issues
    ya_phalaa_issues = []
    for item in missed_opportunities:
        # Check for ya-phalaa in various forms
        if "য়" in item['bengali'] or "য়া" in item['bengali'] or "িয়া" in item['bengali']:
            ya_phalaa_issues.append(item)
    
    print(f"\nNames with ya-phalaa issues: {len(ya_phalaa_issues)}")
    for i, item in enumerate(ya_phalaa_issues[:5]):
        print(f"  {i+1}. '{item['bengali']}' → Simple: '{item['simple']}', Specialized: '{item['specialized']}'")
    
    return {
        'missed_opportunities': missed_opportunities,
        'common_prefixes': common_prefixes,
        'common_suffixes': common_suffixes,
        'ya_phalaa_issues': ya_phalaa_issues
    }

def suggest_pattern_improvements():
    """Suggest improvements to the pattern-based system based on analysis."""
    # First analyze the missed opportunities
    results = analyze_missed_pattern_opportunities()
    
    # Suggest new patterns
    print("\nSuggested pattern improvements:")
    
    # 1. Ya-Phalaa handling
    print("\n1. Improve Ya-Phalaa handling:")
    print("""
   def _normalize_ya_phalaa(text):
       '''Normalize ya-phalaa representations for consistent transliteration.'''
       # Replace য় with য়্ followed by vowel (if any)
       processed_chars = []
       i = 0
       while i < len(text):
           # Check if this is ya-phalaa in its various forms
           if i+1 < len(text) and text[i] == 'য' and text[i+1] == '়':
               processed_chars.append('য়্')  # Mark for special handling
               i += 2  # Skip both characters
           elif text[i] == chr(2527):  # U+09DF ya-phalaa as single character
               processed_chars.append('য়্')  # Mark for special handling
               i += 1
           else:
               processed_chars.append(text[i])
               i += 1
       
       return ''.join(processed_chars)
    """)
    
    # 2. Add more prefix patterns
    print("\n2. Add more prefix patterns:")
    for prefix, items in sorted(results['common_prefixes'].items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        # Calculate most common English translation for this prefix
        english_versions = [item['specialized'].split()[0] if ' ' in item['specialized'] else item['specialized'] for item in items]
        from collections import Counter
        common_english = Counter(english_versions).most_common(1)[0][0] if english_versions else "Unknown"
        
        print(f"   # Pattern for prefix '{prefix}'")
        print(f"   (r\"{prefix}\\s+(.*?)\", lambda match: f\"{common_english} {{transliterate_simple(match.group(1))}}\")")
    
    # 3. Add more suffix patterns
    print("\n3. Add more suffix patterns:")
    for suffix, items in sorted(results['common_suffixes'].items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        # Calculate most common English translation for this suffix
        english_versions = [item['specialized'].split()[-1] if ' ' in item['specialized'] else item['specialized'] for item in items]
        from collections import Counter
        common_english = Counter(english_versions).most_common(1)[0][0] if english_versions else "Unknown"
        
        print(f"   # Pattern for suffix '{suffix}'")
        print(f"   (r\"(.*?)\\s+{suffix}$\", lambda match: f\"{{transliterate_simple(match.group(1))}} {common_english}\")")
    
    # 4. Specific problematic patterns
    print("\n4. Fix specific problematic patterns:")
    for pattern, replacement in [
        ("কাদেরিয়া", "Kaderiya"),
        ("কাদেরীয়া", "Kaderiya"),
        ("কাদরিয়া", "Kaderiya"),
        ("কাদেিয়া", "Kaderiya"),
    ]:
        print(f"   (r\"{pattern}\", \"{replacement}\"),")
    
    # 5. Partial pattern matching
    print("\n5. Implement partial pattern matching:")
    print("""
    def apply_partial_pattern_matches(text):
        '''Apply partial pattern matching for more complex names.'''
        words = text.split()
        
        # If single word, no need for partial matching
        if len(words) <= 1:
            return None
        
        # Try to match patterns on individual words or word groups
        translated_parts = []
        i = 0
        while i < len(words):
            matched = False
            
            # Try matching 2-word combinations first
            if i + 1 < len(words):
                two_word = words[i] + " " + words[i+1]
                result = apply_name_patterns(two_word)
                if result:
                    translated_parts.append(result)
                    i += 2
                    matched = True
                    continue
            
            # Try matching single word
            result = apply_name_patterns(words[i])
            if result:
                translated_parts.append(result)
                matched = True
            else:
                # No pattern match, use simple transliteration
                translated_parts.append(transliterate_simple(words[i]))
                
            i += 1
        
        # Combine the translated parts
        return " ".join(translated_parts)
    """)
    
    return results

def save_suggested_patterns(suggested_patterns, filename="suggested_patterns.json"):
    """Save suggested patterns to a JSON file for reference."""
    # Convert to serializable format
    serializable = {
        'missed_opportunities': [
            {k: v for k, v in item.items()}
            for item in suggested_patterns['missed_opportunities']
        ],
        'common_prefixes': {
            k: [
                {kk: vv for kk, vv in item.items()}
                for item in v
            ]
            for k, v in suggested_patterns['common_prefixes'].items()
        },
        'common_suffixes': {
            k: [
                {kk: vv for kk, vv in item.items()}
                for item in v
            ]
            for k, v in suggested_patterns['common_suffixes'].items()
        },
        'ya_phalaa_issues': [
            {k: v for k, v in item.items()}
            for item in suggested_patterns['ya_phalaa_issues']
        ]
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\nSuggested patterns saved to {filename}")

if __name__ == "__main__":
    # Analyze and suggest improvements
    suggested_patterns = suggest_pattern_improvements()
    
    # Save the suggested patterns for reference
    save_suggested_patterns(suggested_patterns)
