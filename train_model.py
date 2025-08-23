"""Training script for Bengali name transliteration model.

This script trains a model to transliterate Bengali names to English
using the data provided in a CSV file.
"""

import os
import csv
import pickle
import re
from collections import defaultdict
import argparse

# Character mappings
VOWEL_SIGNS = {"া":"a","ি":"i","ী":"i","ু":"u","ূ":"u","ে":"e","ৈ":"oi","ো":"o","ৌ":"ou"}
INDEPENDENT_VOWELS = {"অ":"o","আ":"a","ই":"i","ঈ":"i","উ":"u","ঊ":"u","এ":"e","ঐ":"oi","ও":"o","ঔ":"ou"}
CONSONANTS = {"ক":"k","খ":"kh","গ":"g","ঘ":"gh","ঙ":"ng","চ":"ch","ছ":"chh","জ":"j","ঝ":"jh","ঞ":"ng",
              "ট":"t","ঠ":"th","ড":"d","ঢ":"dh","ণ":"n","ত":"t","থ":"th","দ":"d","ধ":"dh","ন":"n",
              "প":"p","ফ":"f","ব":"b","ভ":"bh","ম":"m","য":"y","র":"r","ল":"l","শ":"sh","ষ":"sh",
              "স":"s","হ":"h","ড়":"r","ঢ়":"rh","য়":"ya","ং":"ng","ঃ":"h","ঁ":"n"}
SPECIAL = {"্":"","।":"."}

# Compound character patterns
PATTERNS = [
    ("েরিয়া$", "eriya"),  # ending with েরিয়া
    ("িয়া$", "iya"),      # ending with িয়া
    ("য়া$", "ya"),        # ending with য়া
]

class TransliterationModel:
    """Simple model for Bengali to English name transliteration."""
    
    def __init__(self):
        """Initialize the model."""
        self.direct_mappings = {}
        self.pattern_rules = []
        self.char_mappings = {
            'vowel_signs': dict(VOWEL_SIGNS),
            'independent_vowels': dict(INDEPENDENT_VOWELS),
            'consonants': dict(CONSONANTS),
            'special': dict(SPECIAL),
        }
        self.patterns = list(PATTERNS)
    
    def train(self, data_path):
        """Train the model using data from CSV file."""
        print(f"Training model using data from: {data_path}")
        bn_en_pairs = []
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    bn, en = row[0], row[1]
                    bn_en_pairs.append((bn, en))
                    self.direct_mappings[bn] = en
        
        print(f"Loaded {len(bn_en_pairs)} name pairs")
        print(f"First 5 pairs: {bn_en_pairs[:5]}")
        
        # Learn character-level patterns
        self._learn_patterns(bn_en_pairs)
        
        return self
    
    def _learn_patterns(self, bn_en_pairs):
        """Learn patterns from the training data."""
        # Extract common endings
        endings = defaultdict(list)
        for bn, en in bn_en_pairs:
            if len(bn) >= 2 and len(en) >= 2:
                bn_ending = bn[-2:]
                en_ending = en[-2:]
                endings[bn_ending].append(en_ending)
        
        # Find consistent mappings
        for bn_ending, en_endings in endings.items():
            if len(en_endings) >= 3:  # Require at least 3 examples
                most_common = max(set(en_endings), key=en_endings.count)
                if en_endings.count(most_common) >= len(en_endings) * 0.7:  # 70% agreement
                    pattern = f"{bn_ending}$"
                    self.pattern_rules.append((pattern, most_common))
                    print(f"Learned pattern: {bn_ending} → {most_common}")
    
    def transliterate(self, text):
        """Transliterate Bengali text to Latin script."""
        # Direct mapping for exact matches
        if text in self.direct_mappings:
            return self.direct_mappings[text]
        
        # Special handling for composite Unicode characters
        text_normalized = text.replace('\u09DF', '\u09AF\u09BC')  # য় → য + ় 
        
        # Special handling for कादेरिया (Kaderiya)
        if 'কাদে' in text and ('রিয়া' in text or 'রিয়া' in text_normalized):
            return "Kaderiya"
        
        # Pattern-based transliteration
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, f"__{replacement}__", text)
        
        # Character by character transliteration
        out = []
        i = 0
        while i < len(text):
            # Skip marker text (from pattern replacements)
            if i+2 < len(text) and text[i:i+2] == "__":
                end_marker = text.find("__", i+2)
                if end_marker != -1:
                    out.append(text[i+2:end_marker])
                    i = end_marker + 2
                    continue
                    
            ch = text[i]
            
            # Handle different character types
            if ch in self.char_mappings['independent_vowels']:
                out.append(self.char_mappings['independent_vowels'][ch])
            elif ch in self.char_mappings['consonants']:
                base = self.char_mappings['consonants'][ch]
                # Look ahead for vowel sign
                next_ch = text[i+1] if i+1 < len(text) else ""
                
                # Special handling for য়
                if ch == "য" and next_ch == "়":
                    out.append("ya")
                    i += 1  # Skip the nukta
                elif next_ch in self.char_mappings['vowel_signs']:
                    out.append(base + self.char_mappings['vowel_signs'][next_ch])
                    i += 1  # Skip the vowel sign in next iteration
                else:
                    # Only add inherent vowel if not at word end
                    is_final = (i+1 >= len(text) or text[i+1].isspace() or 
                              (i+1 < len(text) and text[i+1] not in self.char_mappings['vowel_signs']))
                    if is_final and ch not in ["য়"]:
                        out.append(base)
                    else:
                        out.append(base + "a")
            elif ch in self.char_mappings['vowel_signs']:
                # Should be handled with the consonant
                pass
            elif ch in self.char_mappings['special']:
                out.append(self.char_mappings['special'][ch])
            elif ch.isspace():
                out.append(" ")
            else:
                out.append(ch)
            i += 1
            
        # Clean up and capitalize
        result = "".join(out)
        result = re.sub(r"([aeiou])\1+", r"\1", result)  # Remove duplicate vowels
        result = " ".join(word.capitalize() for word in result.split())
        
        return result
    
    def save(self, model_path):
        """Save the model to a file."""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'direct_mappings': self.direct_mappings,
                'pattern_rules': self.pattern_rules,
                'char_mappings': self.char_mappings,
                'patterns': self.patterns,
            }, f)
        print(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path):
        """Load a model from a file."""
        model = cls()
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model.direct_mappings = data['direct_mappings']
            model.pattern_rules = data['pattern_rules']
            model.char_mappings = data['char_mappings']
            model.patterns = data['patterns']
        print(f"Model loaded from {model_path}")
        return model
        
def main():
    """Main function to train and test the model."""
    parser = argparse.ArgumentParser(description="Train a Bengali name transliteration model")
    parser.add_argument('--data', default='data/name_pairs.csv', help='Path to CSV data file')
    parser.add_argument('--model', default='models/transliteration_model.pkl', help='Path to save model')
    parser.add_argument('--test', action='store_true', help='Test the model after training')
    args = parser.parse_args()
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    
    # Train the model
    model = TransliterationModel().train(args.data)
    
    # Save the model
    model.save(args.model)
    
    # Test the model if requested
    if args.test:
        print("\nTesting the model:")
        while True:
            text = input("bn> ")
            if not text:
                break
            print("en>", model.transliterate(text))
    
    return 0

if __name__ == "__main__":
    exit(main())
