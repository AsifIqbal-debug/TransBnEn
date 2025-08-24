# Test file for comprehensive Bengali name transliteration

import csv
import os
from TransBnEn.translateIndigo_with_name import (
    transliterate, transliterate_name_specialized, 
    transliterate_simple, apply_name_patterns,
    NAME_MAPPINGS, SURNAME_MAPPINGS, PREFIX_MAPPINGS, 
    FEMALE_SUFFIX_MAPPINGS, NAME_CONNECTORS, init_name_patterns
)

# Initialize the patterns
init_name_patterns()

def generate_test_names(count=200):
    """Generate a diverse set of test names by combining components."""
    names = []
    
    # First, add all direct mappings we have
    for bn_name in NAME_MAPPINGS:
        if len(bn_name.split()) <= 3:  # Keep names reasonably short
            names.append(bn_name)
    
    # If we need more, start combining components
    prefixes = list(PREFIX_MAPPINGS.keys())[:10]
    surnames = list(SURNAME_MAPPINGS.keys())[:20]
    female_suffixes = list(FEMALE_SUFFIX_MAPPINGS.keys())[:10]
    connectors = list(NAME_CONNECTORS.keys())[:5]
    
    # Simple names: prefix + surname
    for prefix in prefixes:
        for surname in surnames[:5]:
            names.append(f"{prefix} {surname}")
    
    # Female names: prefix + female suffix
    for prefix in prefixes[:5]:
        for suffix in female_suffixes[:5]:
            names.append(f"{prefix} {suffix}")
    
    # Complex names: prefix + connector + surname
    for prefix in prefixes[:3]:
        for connector in connectors[:2]:
            for surname in surnames[:3]:
                names.append(f"{prefix} {connector} {surname}")
    
    # Special test cases for problematic names
    special_cases = [
        "আমায়রা খাতুন",
        "আমায়রা ইকবাল",
        "মোঃ খুশি মিয়া",
        "শ্রী অম্বী চন্দ্র সরকার",
        "কাদেরিয়া",
        "কাদেরীয়া",
        "কাদরিয়া",
        "মোঃ আবেদীন",
        "মোছাঃ রুমা খাতুন",
        "শেখ হাসিনা",
        "জনাব করিম উদ্দিন",
        "ড. মুহাম্মদ ইউনূস",
        "প্রফেসর আবদুল করিম",
        "শাহ আলম খান",
        "নূর জাহান বেগম",
        "রহিম চৌধুরী",
        "ফাতেমা সুলতানা"
    ]
    
    names.extend(special_cases)
    
    # Ensure we have unique names and limit to the requested count
    unique_names = list(set(names))
    return unique_names[:count]

def test_transliteration(names):
    """Test transliteration methods on a list of names."""
    results = []
    
    print(f"Testing {len(names)} Bengali names with different transliteration methods...")
    
    for idx, name in enumerate(names):
        # Apply different transliteration methods
        simple_result = transliterate_simple(name)
        specialized_result = transliterate_name_specialized(name)
        pattern_result = apply_name_patterns(name) or "No pattern match"
        general_result = transliterate(name)
        
        # Track differences between methods
        methods_match = (simple_result == specialized_result == general_result)
        
        results.append({
            'index': idx + 1,
            'bengali_name': name,
            'simple': simple_result,
            'specialized': specialized_result, 
            'pattern': pattern_result,
            'general': general_result,
            'methods_match': methods_match
        })
        
    return results

def print_results(results):
    """Print test results in a formatted way."""
    print(f"\nTransliteration results for {len(results)} names:")
    print("-" * 100)
    print(f"{'#':4} | {'Bengali Name':30} | {'Simple':20} | {'Specialized':20} | {'Pattern':20} | {'General':20} | {'Match'}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['index']:4} | {r['bengali_name']:30} | {r['simple']:20} | {r['specialized']:20} | {r['pattern']:20} | {r['general']:20} | {r['methods_match']}")
    
    # Summary stats
    match_count = sum(1 for r in results if r['methods_match'])
    pattern_match_count = sum(1 for r in results if r['pattern'] != "No pattern match")
    
    print("-" * 100)
    print(f"Summary: {match_count}/{len(results)} names ({match_count/len(results)*100:.1f}%) have consistent transliteration across methods")
    print(f"Pattern matches: {pattern_match_count}/{len(results)} names ({pattern_match_count/len(results)*100:.1f}%) matched a pattern")

def save_to_csv(results, filename="transliteration_results.csv"):
    """Save results to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'index', 'bengali_name', 'simple', 'specialized', 
            'pattern', 'general', 'methods_match'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {os.path.abspath(filename)}")

if __name__ == "__main__":
    # Generate test names
    test_names = generate_test_names(200)
    print(f"Generated {len(test_names)} unique test names")
    
    # Run the transliteration tests
    results = test_transliteration(test_names)
    
    # Print results
    print_results(results)
    
    # Save results to CSV
    save_to_csv(results)
