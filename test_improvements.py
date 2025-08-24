# Test improvements to the pattern-based transliteration system

import csv
from TransBnEn.translateIndigo_with_name import (
    transliterate, transliterate_name_specialized, 
    transliterate_simple, apply_name_patterns,
    NAME_PATTERNS, init_name_patterns
)

# Initialize the patterns
init_name_patterns()

def test_ya_phalaa_fixes():
    """Test fixes for ya-phalaa issues."""
    test_cases = [
        "আমায়রা",
        "আমায়রা খাতুন",
        "আমায়রা ইকবাল",
        "আয়েশা",
        "সায়েদা",
        "জায়েদা",
        "রেজোয়ান",
        "রিয়াজ",
        "শায়লা",
        "শায়মা",
        "কাদেরিয়া",
        "কাদেরীয়া",
        "কাদরিয়া"
    ]
    
    print("\nTesting Ya-Phalaa Fixes:")
    print(f"{'Bengali Name':20} | {'Pattern Result':20} | {'General Result':20}")
    print("-" * 65)
    
    for name in test_cases:
        pattern_result = apply_name_patterns(name) or "No pattern match"
        general_result = transliterate(name)
        match = pattern_result == general_result or (
            pattern_result != "No pattern match" and general_result.replace(" ", "") == pattern_result.replace(" ", "")
        )
        
        print(f"{name:20} | {pattern_result:20} | {general_result:20} | {'✅' if match else '❌'}")

def test_partial_pattern_matching():
    """Test partial pattern matching for complex names."""
    test_cases = [
        "মোঃ আসিফ রহমান",
        "শেখ হাসিনা বেগম",
        "জনাব করিম উদ্দিন",
        "শ্রী বিমল চন্দ্র সরকার",
        "মোঃ আবেদীন ইসলাম",
        "ড. রফিক আলী",
        "মোছাঃ নাসরিন খাতুন",
        "শেখ মুজিবুর রহমান",
        "ড. মুহাম্মদ ইউনূস",
        "প্রফেসর আবদুল করিম"
    ]
    
    print("\nTesting Partial Pattern Matching:")
    print(f"{'Bengali Name':25} | {'Pattern Result':25} | {'General Result':25}")
    print("-" * 80)
    
    for name in test_cases:
        pattern_result = apply_name_patterns(name) or "No pattern match"
        general_result = transliterate(name)
        match = pattern_result == general_result or (
            pattern_result != "No pattern match" and general_result.replace(" ", "") == pattern_result.replace(" ", "")
        )
        
        print(f"{name:25} | {pattern_result:25} | {general_result:25} | {'✅' if match else '❌'}")

def run_comparative_test():
    """Compare performance between old and new implementations."""
    # We can't actually run a true comparison without the old code,
    # but we can check if our changes improved pattern match rates
    
    # Load test names from CSV if available
    test_names = []
    try:
        with open("transliteration_results_500.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_names.append(row['bengali_name'])
    except Exception:
        # Use a smaller set of test names
        test_names = [
            "মোঃ আসিফ",
            "শেখ হাসিনা",
            "মোছাঃ রুমা খাতুন",
            "ড. মুহাম্মদ ইউনূস",
            "মোঃ খুশি মিয়া",
            "শ্রী অম্বী চন্দ্র সরকার",
            "আমায়রা খাতুন",
            "জনাব করিম উদ্দিন",
            "আয়েশা বেগম",
            "সায়েদা খাতুন",
            "জায়েদা আক্তার",
            "রেজোয়ান হক",
            "রিয়াজ আহমেদ",
            "শায়লা",
            "শায়মা খাতুন",
            "কাদেরিয়া",
            "কাদেরীয়া",
            "কাদরিয়া",
            "মোঃ আবেদীন ইসলাম",
            "শেখ মুজিবুর রহমান"
        ]
    
    match_count = 0
    results = []
    
    print("\nTesting Pattern Matching Performance:")
    for idx, name in enumerate(test_names):
        pattern_result = apply_name_patterns(name) or "No pattern match"
        general_result = transliterate(name)
        
        if pattern_result != "No pattern match":
            match_count += 1
        
        results.append({
            'bengali_name': name,
            'pattern': pattern_result,
            'general': general_result
        })
    
    print(f"Pattern match rate: {match_count}/{len(test_names)} ({match_count/len(test_names)*100:.1f}%)")
    
    # Print sample of results
    print(f"\nSample results (first 10):")
    print(f"{'Bengali Name':25} | {'Pattern Result':25} | {'General Result':25}")
    print("-" * 80)
    for r in results[:10]:
        print(f"{r['bengali_name']:25} | {r['pattern']:25} | {r['general']:25}")

if __name__ == "__main__":
    print("Testing Pattern-Based Transliteration Improvements")
    print("=" * 50)
    
    # Test ya-phalaa fixes
    test_ya_phalaa_fixes()
    
    # Test partial pattern matching
    test_partial_pattern_matching()
    
    # Run comparative test
    run_comparative_test()
