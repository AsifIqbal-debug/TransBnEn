# Script to analyze pattern-based transliteration performance

import csv
import os
from collections import Counter
import re
from pathlib import Path
from TransBnEn.translateIndigo_with_name import (
    transliterate, transliterate_name_specialized, 
    apply_name_patterns, init_name_patterns,
    NAME_PATTERNS
)

# Initialize the patterns
init_name_patterns()

def analyze_pattern_matches(csv_path):
    """Analyze pattern matches from test results CSV."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    # Count pattern matches
    pattern_matches = [r for r in results if r['pattern'] != "No pattern match"]
    
    # Analyze which patterns matched the most
    matched_patterns = []
    for r in pattern_matches:
        bengali_name = r['bengali_name']
        for i, (pattern, _) in enumerate(NAME_PATTERNS):
            if re.search(pattern, bengali_name):
                matched_patterns.append(f"Pattern {i+1}")
                break
    
    pattern_counts = Counter(matched_patterns)
    
    print("\nPattern Matching Analysis:")
    print("-" * 60)
    print(f"Total names tested: {len(results)}")
    print(f"Names with pattern match: {len(pattern_matches)} ({len(pattern_matches)/len(results)*100:.1f}%)")
    
    print("\nTop matching patterns:")
    for pattern, count in pattern_counts.most_common(10):
        print(f"  {pattern}: {count} matches ({count/len(results)*100:.1f}%)")
    
    # Analyze common prefixes/suffixes in pattern matches
    prefix_matches = []
    for r in pattern_matches:
        bengali_name = r['bengali_name']
        if bengali_name.startswith("মোঃ "):
            prefix_matches.append("মোঃ")
        elif bengali_name.startswith("শ্রী "):
            prefix_matches.append("শ্রী")
        elif bengali_name.startswith("মোছাঃ "):
            prefix_matches.append("মোছাঃ")
        elif bengali_name.startswith("ড. "):
            prefix_matches.append("ড.")
        elif bengali_name.startswith("প্রফে. "):
            prefix_matches.append("প্রফে.")
        elif bengali_name.startswith("শেখ "):
            prefix_matches.append("শেখ")
    
    # Count suffix matches
    suffix_matches = []
    for r in pattern_matches:
        bengali_name = r['bengali_name']
        if bengali_name.endswith(" বেগম"):
            suffix_matches.append("বেগম")
        elif bengali_name.endswith(" খাতুন"):
            suffix_matches.append("খাতুন")
        elif bengali_name.endswith(" ইসলাম"):
            suffix_matches.append("ইসলাম")
        elif bengali_name.endswith(" রহমান"):
            suffix_matches.append("রহমান")
        elif bengali_name.endswith(" হোসেন"):
            suffix_matches.append("হোসেন")
        elif bengali_name.endswith(" আলী"):
            suffix_matches.append("আলী")
        elif bengali_name.endswith(" উদ্দিন"):
            suffix_matches.append("উদ্দিন")
    
    prefix_counts = Counter(prefix_matches)
    suffix_counts = Counter(suffix_matches)
    
    print("\nTop matched prefixes:")
    for prefix, count in prefix_counts.most_common(5):
        print(f"  {prefix}: {count} matches")
    
    print("\nTop matched suffixes:")
    for suffix, count in suffix_counts.most_common(5):
        print(f"  {suffix}: {count} matches")
    
    # Analyze problematic patterns (cases where pattern match but inconsistent results)
    problematic = [r for r in pattern_matches if r['pattern'] != r['general']]
    
    print(f"\nProblematic pattern matches (pattern ≠ general): {len(problematic)}/{len(pattern_matches)} ({len(problematic)/len(pattern_matches)*100:.1f}%)")
    
    # Show examples of problematic pattern matches
    if problematic:
        print("\nExamples of problematic matches:")
        for i, r in enumerate(problematic[:5]):
            print(f"  {i+1}. '{r['bengali_name']}' → Pattern: '{r['pattern']}', General: '{r['general']}'")
    
    # Calculate consistency between methods for pattern matches
    pattern_consistency = [r for r in pattern_matches if r['methods_match']]
    print(f"\nConsistency for pattern matches: {len(pattern_consistency)}/{len(pattern_matches)} ({len(pattern_consistency)/len(pattern_matches)*100:.1f}%)")
    
    # Test with direct application of pattern-based transliteration
    print("\nDirect pattern transliteration test:")
    test_cases = [
        "মোঃ আসিফ",
        "শেখ হাসিনা",
        "মোছাঃ রুমা খাতুন",
        "ড. মুহাম্মদ ইউনূস",
        "মোঃ খুশি মিয়া",
        "শ্রী অম্বী চন্দ্র সরকার",
        "আমায়রা খাতুন",
        "জনাব করিম উদ্দিন"
    ]
    
    print(f"{'Bengali Name':30} | {'Pattern Result':25} | {'General Result':25}")
    print("-" * 80)
    for name in test_cases:
        pattern_result = apply_name_patterns(name) or "No match"
        general_result = transliterate(name)
        print(f"{name:30} | {pattern_result:25} | {general_result:25}")

if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent / "transliteration_results_500.csv"
    analyze_pattern_matches(csv_path)
