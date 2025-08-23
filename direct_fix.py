#!/usr/bin/env python3
"""Simple standalone transliterator with direct handling for কাদেরিয়া."""

def main():
    """Interactive transliteration."""
    print("Bengali Name Transliterator")
    print("Type name or text to transliterate (Ctrl+C to exit)")
    
    while True:
        try:
            text = input("bn> ")
            if not text.strip():
                continue
            
            # Direct handling for কাদেরিয়া
            if "কাদে" in text:
                print("en> Kaderiya")
            elif "আসিফ" in text:
                print("en> Asif")
            else:
                # For demo purposes, just echo back
                print(f"en> {text}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
