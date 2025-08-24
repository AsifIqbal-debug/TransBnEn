"""Test the transliteration of Bengali surnames as postfixes."""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent))

# Import the transliteration functions
try:
    from TransBnEn.translateIndigo_with_name import transliterate, transliterate_name_specialized
except ImportError:
    print("Failed to import transliteration functions. Make sure the project is in your PYTHONPATH.")
    sys.exit(1)

def test_surname_postfix():
    """Test the transliteration of names with common Bengali surnames."""
    # Test cases with format: (Bengali name, Expected English transliteration)
    test_cases = [
        # Title + Name + Surname format
        ("শ্রী অভিজিৎ চক্রবর্তী", "Shri Abhijit Chakraborty"),
        ("ড. সুমন চট্টোপাধ্যায়", "Dr. Suman Chatterjee"),
        ("মোঃ আরিফ খান", "Md. Arif Khan"),
        ("প্রফে. অনিল সরকার", "Prof. Anil Sarkar"),
        
        # Name + Surname format
        ("অমল বসু", "Amal Basu"),
        ("দীপক গুপ্ত", "Dipak Gupta"),
        ("রাজেশ মুখোপাধ্যায়", "Rajesh Mukherjee"),
        ("সুদীপ্ত বন্দ্যোপাধ্যায়", "Sudipta Banerjee"),
        ("শুভম ভট্টাচার্য", "Shubham Bhattacharya"),
        ("তনুশ্রী সিংহ", "Tanushree Sinha"),
        ("আশিক রহমান", "Ashik Rahman"),
        ("কামরুল ইসলাম", "Kamrul Islam"),
        ("সমীর দাস", "Samir Das"),
        ("সুমন দত্ত", "Suman Datta"),
        ("নিখিল গুহ", "Nikhil Guha"),
        ("প্রতাপ রায়", "Pratap Ray"),
        ("অনির্বান সেন", "Anirban Sen"),
        
        # Names with more complex structures
        ("মোঃ আবদুল করিম খান", "Md. Abdul Karim Khan"),
        ("শ্রীমতী কল্যাণী দেবী চট্টোপাধ্যায়", "Srimoti Kalyani Devi Chatterjee"),
        ("সৈয়দ নুরুল হোসেন", "Syed Nurul Hossain"),
        ("ড. অরিন্দম মুখোপাধ্যায়", "Dr. Arindam Mukherjee")
    ]
    
    # Run the tests and display results
    print("Testing Bengali surname postfix transliterations:")
    print("-" * 70)
    
    passed = 0
    failed = 0
    
    for bn_name, expected_en in test_cases:
        # Test with both transliteration functions
        result_trans = transliterate(bn_name)
        result_specialized = transliterate_name_specialized(bn_name)
        
        # Check if either function gives the correct result
        if result_trans == expected_en or result_specialized == expected_en:
            passed += 1
            print(f"✅ PASS: {bn_name} → {expected_en}")
            if result_trans != result_specialized:
                print(f"   Note: transliterate: '{result_trans}' | specialized_name: '{result_specialized}'")
        else:
            failed += 1
            print(f"❌ FAIL: {bn_name}")
            print(f"   Expected: {expected_en}")
            print(f"   Got (trans): {result_trans}")
            print(f"   Got (specialized): {result_specialized}")
    
    print("-" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    
    return passed, failed

if __name__ == "__main__":
    passed, failed = test_surname_postfix()
    sys.exit(1 if failed else 0)
