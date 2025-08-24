# Test file for comprehensive Bengali name transliteration (500 names)

import csv
import os
import random
from pathlib import Path
from TransBnEn.translateIndigo_with_name import (
    transliterate, transliterate_name_specialized, 
    transliterate_simple, apply_name_patterns,
    NAME_MAPPINGS, SURNAME_MAPPINGS, PREFIX_MAPPINGS, 
    FEMALE_SUFFIX_MAPPINGS, NAME_CONNECTORS, init_name_patterns
)

# Initialize the patterns
init_name_patterns()

def load_name_data():
    """Load name data from available CSV files."""
    name_pairs = []
    
    # Try to load from name_pairs.csv
    name_pairs_path = Path(__file__).resolve().parent / "data" / "name_pairs.csv"
    if name_pairs_path.exists():
        try:
            with open(name_pairs_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'Bengali' in row and 'English' in row:
                        name_pairs.append((row['Bengali'], row['English']))
        except Exception as e:
            print(f"Error loading name_pairs.csv: {e}")
    
    # Try to load from custom_names.csv
    custom_names_path = Path(__file__).resolve().parent / "TransBnEn" / "custom_names.csv"
    if custom_names_path.exists():
        try:
            with open(custom_names_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'bangla' in row and 'english' in row and row['bangla']:
                        # Only add if it looks like a name (no punctuation)
                        if not any(p in row['bangla'] for p in ".,?!-"):
                            name_pairs.append((row['bangla'], row['english']))
        except Exception as e:
            print(f"Error loading custom_names.csv: {e}")
    
    return name_pairs

def generate_test_names(count=500):
    """Generate a diverse set of test names by combining components and loading from files."""
    names = []
    
    # Load name pairs from files
    name_pairs = load_name_data()
    
    # Add Bengali names from loaded pairs
    for bn_name, _ in name_pairs:
        # Only add if it's a plausible name (not a sentence)
        if len(bn_name.split()) <= 4 and not any(p in bn_name for p in ".,?!-"):
            names.append(bn_name)
    
    # Add all direct mappings we have
    for bn_name in NAME_MAPPINGS:
        if len(bn_name.split()) <= 3:  # Keep names reasonably short
            names.append(bn_name)
    
    # If we need more, start combining components
    prefixes = list(PREFIX_MAPPINGS.keys())
    surnames = list(SURNAME_MAPPINGS.keys())
    female_suffixes = list(FEMALE_SUFFIX_MAPPINGS.keys())
    connectors = list(NAME_CONNECTORS.keys())
    
    # Simple names: prefix + surname
    for prefix in prefixes[:15]:
        for surname in surnames[:15]:
            names.append(f"{prefix} {surname}")
    
    # Female names: prefix + female suffix
    for prefix in prefixes[:10]:
        for suffix in female_suffixes[:10]:
            names.append(f"{prefix} {suffix}")
    
    # Complex names: prefix + connector + surname
    for prefix in prefixes[:5]:
        for connector in connectors[:3]:
            for surname in surnames[:5]:
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
        "ফাতেমা সুলতানা",
        "সায়মা হোসেন",
        "তানভীর আহমেদ",
        "নাজমুল হাসান",
        "সাবিনা ইয়াসমিন",
        "কামরুল হাসান",
        "নাসরিন আক্তার",
        "আশরাফুল আলম",
        "শাহীন আলম",
        "রাশেদা বেগম",
        "আনিসুর রহমান",
        "মাহমুদুল হাসান",
        "জসিম উদ্দিন",
        "শাহনাজ পারভীন",
        "সাদিয়া আফরিন",
        "তানজিম আনজুম",
        "শামীম আরা",
        "নাজমুল হুদা",
        "শাকিল আহমেদ",
        "রাফি খান",
        "নাসিমা আক্তার",
        "জাকির হোসেন",
        "নাদিয়া সুলতানা",
        "মাসুদ রানা"
    ]
    
    names.extend(special_cases)
    
    # Ensure we have unique names
    unique_names = list(set(names))
    
    # If we still don't have enough names, create more by combining existing names
    if len(unique_names) < count:
        # Generate composite names by combining parts of existing names
        existing_parts = []
        for name in unique_names:
            parts = name.split()
            existing_parts.extend(parts)
        
        # Create unique parts
        unique_parts = list(set(existing_parts))
        
        # Generate additional random combinations
        while len(unique_names) < count:
            num_parts = random.randint(2, 3)  # Generate names with 2-3 parts
            if num_parts <= len(unique_parts):
                new_name = " ".join(random.sample(unique_parts, num_parts))
                unique_names.append(new_name)
    
    # Shuffle and limit to requested count
    random.shuffle(unique_names)
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
    print("-" * 120)
    print(f"{'#':4} | {'Bengali Name':30} | {'Simple':20} | {'Specialized':20} | {'Pattern':20} | {'General':20} | {'Match'}")
    print("-" * 120)
    
    # Only print the first 20 results to avoid excessive output
    for r in results[:20]:
        print(f"{r['index']:4} | {r['bengali_name']:30} | {r['simple']:20} | {r['specialized']:20} | {r['pattern'][:20]:20} | {r['general']:20} | {r['methods_match']}")
    
    if len(results) > 20:
        print("... (output truncated for brevity)")
    
    # Summary stats
    match_count = sum(1 for r in results if r['methods_match'])
    pattern_match_count = sum(1 for r in results if r['pattern'] != "No pattern match")
    
    print("-" * 120)
    print(f"Summary: {match_count}/{len(results)} names ({match_count/len(results)*100:.1f}%) have consistent transliteration across methods")
    print(f"Pattern matches: {pattern_match_count}/{len(results)} names ({pattern_match_count/len(results)*100:.1f}%) matched a pattern")
    
    # Calculate method-specific statistics
    simple_specialized_match = sum(1 for r in results if r['simple'] == r['specialized'])
    simple_general_match = sum(1 for r in results if r['simple'] == r['general'])
    specialized_general_match = sum(1 for r in results if r['specialized'] == r['general'])
    
    print(f"Simple = Specialized: {simple_specialized_match}/{len(results)} ({simple_specialized_match/len(results)*100:.1f}%)")
    print(f"Simple = General: {simple_general_match}/{len(results)} ({simple_general_match/len(results)*100:.1f}%)")
    print(f"Specialized = General: {specialized_general_match}/{len(results)} ({specialized_general_match/len(results)*100:.1f}%)")

def save_to_csv(results, filename="transliteration_results_500.csv"):
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
    test_names = generate_test_names(500)
    print(f"Generated {len(test_names)} unique test names")
    
    # Run the transliteration tests
    results = test_transliteration(test_names)
    
    # Print results
    print_results(results)
    
    # Save results to CSV
    save_to_csv(results)
