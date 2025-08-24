"""Test all Bengali title and prefix transliterations."""

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

# Define test cases for all prefixes/titles
test_cases = [
    # Title/prefix alone
    ("শ্রী", "Shri"),
    ("শ্রীযুক্ত", "Srijukto"),
    ("শ্রীমতী", "Srimoti"),
    ("কুমারী", "Kumari"),
    ("সুশ্রী", "Sushri"),
    ("জনাব", "Janab"),
    ("জনাবা", "Janaba"),
    ("মোঃ", "Md."),
    ("হাজী", "Haji"),
    ("আলহাজ্", "Al-Haj"),
    ("মাওলানা", "Maulana"),
    ("মরহুম", "Marhum"),
    ("মরহুমা", "Marhuma"),
    ("প্রয়াত", "Prayat"),
    ("স্বর্গীয়", "Swargiyo"),
    ("ড.", "Dr."),
    ("প্রফে.", "Prof."),
    ("ইঞ্জি.", "Engr."),
    ("অ্যাড.", "Adv."),
    ("মোছাঃ", "Mst."),
    ("মোছা", "Mst."),
    
    # With names
    ("শ্রী রমেশ চন্দ্র", "Shri Ramesh Chandra"),
    ("শ্রীযুক্ত সুনীল দত্ত", "Srijukto Sunil Dutta"),
    ("শ্রীমতী সুমিতা দাস", "Srimoti Sumita Das"),
    ("কুমারী শিল্পী রায়", "Kumari Shilpi Ray"),
    ("সুশ্রী অনামিকা ঘোষ", "Sushri Anamika Ghosh"),
    ("জনাব আবদুল করিম", "Janab Abdul Karim"),
    ("জনাবা শাহীনা বেগম", "Janaba Shahina Begum"),
    ("মোঃ আসিফ ইকবাল", "Md. Asif Iqbal"),
    ("হাজী মোহাম্মদ আলী", "Haji Mohammad Ali"),
    ("আলহাজ্ জাফর আলী", "Al-Haj Jafar Ali"),
    ("মাওলানা নুরুল ইসলাম", "Maulana Nurul Islam"),
    ("মরহুম আব্দুর রহমান", "Marhum Abdur Rahman"),
    ("মরহুমা ফাতেমা বেগম", "Marhuma Fatema Begum"),
    ("প্রয়াত সত্যজিৎ রায়", "Prayat Satyajit Ray"),
    ("স্বর্গীয় অভিজিৎ সেন", "Swargiyo Abhijit Sen"),
    ("ড. তানজিনা আফরোজ", "Dr. Tanjina Afroz"),
    ("প্রফে. মাহমুদুল হাসান", "Prof. Mahmudul Hasan"),
    ("ইঞ্জি. শাহীন আলম", "Engr. Shahin Alam"),
    ("অ্যাড. জাহাঙ্গীর আলম", "Adv. Jahangir Alam"),
    ("মোছাঃ তহমিনা বেগম", "Mst. Tahamina Begum"),
    
    # With variations (extra spaces, quotes)
    ("'শ্রী রমেশ চন্দ্র'", "Shri Ramesh Chandra"),
    ("  শ্রীমতী  সুমিতা  দাস  ", "Srimoti Sumita Das"),
    ("মোঃ  আসিফ  ইকবাল", "Md. Asif Iqbal"),
    ("মোছাঃ   তহমিনা   বেগম", "Mst. Tahamina Begum"),
]

def run_tests():
    """Run all test cases for transliteration."""
    passed = 0
    failed = 0
    
    print("Testing Bengali title/prefix transliterations...")
    print("-" * 60)
    
    for bn_text, expected_en in test_cases:
        # Test with transliterate function
        result_trans = transliterate(bn_text)
        
        # Test with specialized name transliteration
        result_name = transliterate_name_specialized(bn_text)
        
        # Check if either function gives the correct result
        if result_trans == expected_en or result_name == expected_en:
            passed += 1
            print(f"✅ PASS: {bn_text} → {expected_en}")
            if result_trans != result_name:
                print(f"   Note: transliterate: '{result_trans}' | specialized_name: '{result_name}'")
        else:
            failed += 1
            print(f"❌ FAIL: {bn_text}")
            print(f"   Expected: {expected_en}")
            print(f"   Got (trans): {result_trans}")
            print(f"   Got (specialized): {result_name}")
        
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    return passed, failed

if __name__ == "__main__":
    passed, failed = run_tests()
    sys.exit(1 if failed else 0)
