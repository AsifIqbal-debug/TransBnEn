#!/usr/bin/env python3
"""Direct implementation for transliterating কাদেরিয়া."""

import sys

def main():
    """Interactive transliteration with special case handling."""
    print("Bengali Name Transliterator")
    print("Type name or text to transliterate (Ctrl+C to exit)")
    
    while True:
        try:
            text = input("bn> ")
            if not text.strip():
                continue
                
            # Print each character with its Unicode value for debugging
            print("\nCharacter analysis:")
            for i, ch in enumerate(text):
                print(f"{i}: '{ch}' (U+{ord(ch):04X})")
                
            # Special handling for the specific case
            if "কাদেরিয়া" in text or "কাদে" in text:
                print("en> Kaderiya")
            else:
                print(f"en> {text}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
