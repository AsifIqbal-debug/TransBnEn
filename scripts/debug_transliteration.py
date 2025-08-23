"""Debug script for Bengali transliteration.

This script helps analyze how each character in a Bengali text 
is being transliterated to Latin script.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TransBnEn.transliterate import transliterate
import re

def debug_transliteration(text):
    """Helper function to debug transliteration issues."""
    print(f"Input: {text}")
    print(f"Output: {transliterate(text)}")
    
    # Character-by-character analysis
    print("\nCharacter analysis:")
    for i, ch in enumerate(text):
        # Determine the character type
        char_type = "Unknown"
        if 0x0980 <= ord(ch) <= 0x09FF:  # Bengali Unicode block
            if ch in "অআইঈউঊএঐওঔ":
                char_type = "Independent Vowel"
            elif ch in "ািীুূেৈোৌ":
                char_type = "Vowel Sign"
            elif ch in "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়":
                char_type = "Consonant"
            elif ch in "্ং":
                char_type = "Special Sign"
            else:
                char_type = "Bengali"
        
        print(f"{i}: '{ch}' (Unicode: U+{ord(ch):04X}) - {char_type}")
    
    # Pattern detection
    print("\nPattern detection:")
    patterns = [
        ("েরিয়া$", "ends with েরিয়া"),
        ("িয়া$", "ends with িয়া"),
        ("য়া$", "ends with য়া"),
    ]
    
    for pattern, desc in patterns:
        if re.search(pattern, text):
            print(f"Matches: {desc}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_transliteration(sys.argv[1])
    else:
        print("Bengali Transliteration Debugger")
        print("Enter Bengali text to analyze (Ctrl+C to exit)")
        
        while True:
            try:
                text = input("\nbn> ")
                if not text.strip():
                    continue
                    
                debug_transliteration(text)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
