#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Bengali-English transliteration with both prefixes and suffixes.
This tests the handling of names that have both title prefixes and surname postfixes.
"""

import sys
from pathlib import Path
import os

# Add the parent directory to sys.path to import from TransBnEn
script_path = Path(__file__).resolve().parent
sys.path.append(str(script_path))

from TransBnEn.translateIndigo_with_father_name import transliterate

def test_combined_prefix_suffix():
    """Test the transliteration of names with both prefixes and suffixes."""
    
    # Test cases with both prefixes and suffixes
    test_cases = [
        # Format: (Bengali name, Expected English transliteration)
        ("শ্রী অভিজিৎ চক্রবর্তী", "Shri Abhijit Chakraborty"),
        ("শ্রীমতী কল্যাণী দেবী চট্টোপাধ্যায়", "Srimoti Kalyani Devi Chatterjee"),
        ("ডাঃ সুব্রত বসু", "Dr. Subrata Basu"),
        ("প্রফেসর দেবাশিস মুখার্জী", "Professor Debashis Mukherjee"),
        ("ডক্টর সুনীল গুপ্ত", "Doctor Sunil Gupta"),
        ("মোঃ জাহাঙ্গীর খান", "Md. Jahangir Khan"),
        ("জনাব আরিফুল ইসলাম", "Janab Ariful Islam"),
        ("শেখ মোস্তাফিজুর রহমান", "Sheikh Mostafizur Rahman"),
        ("ক্যাপ্টেন আবদুল করিম", "Captain Abdul Karim"),
        ("মাওলানা আবদুর রহিম খন্দকার", "Maulana Abdur Rahim Khandaker"),
        ("অধ্যাপক শঙ্কর রায়", "Adjapok Shankar Ray"),
        ("স্বর্গীয় সুকুমার সেন", "Swargiyo Sukumar Sen"),
        ("আলহাজ্ নূর মোহাম্মদ হোসেন", "Al-Haj Nur Mohammad Hossain"),
        ("বেগম হাসিনা খাতুন", "Begum Hasina Khatun"),
        ("রাণী কুমুদিনী দেবী", "Rani Kumudini Devi")
    ]
    
    # Run the tests
    print("Testing names with both prefixes and suffixes...")
    passed = 0
    failed = 0
    
    for i, (bengali, expected) in enumerate(test_cases, 1):
        result = transliterate(bengali)
        if result == expected:
            print(f"✅ Test {i} passed: {bengali} → {result}")
            passed += 1
        else:
            print(f"❌ Test {i} failed: {bengali}")
            print(f"   Expected: {expected}")
            print(f"   Got     : {result}")
            failed += 1
    
    # Print summary
    total = passed + failed
    print(f"\nSummary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return passed == total

if __name__ == "__main__":
    success = test_combined_prefix_suffix()
    sys.exit(0 if success else 1)
