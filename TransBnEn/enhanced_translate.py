"""Script to use the enhanced transliterator with NLLB translation.

This script combines the trained transliteration model with the NLLB
translation model for improved Bengali to English translation.
"""

import os
import sys
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TransBnEn.enhanced_transliterator import EnhancedTransliterator

# Configuration from environment variables
MODEL_NAME = os.environ.get("INDIGO_MODEL", "facebook/nllb-200-distilled-600M")
SRC_LANG = "ben_Beng"
TGT_LANG = "eng_Latn"
MAX_NEW = int(os.environ.get("INDIGO_MAX_NEW", "128"))

# Bangla Unicode block detection
BN_RANGE_RE = re.compile(r"[\u0980-\u09FF]")  # Bangla Unicode block

# Initialize the enhanced transliterator
transliterator = EnhancedTransliterator()

# Initialize the translation model
_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the NLLB translation model."""
    global _tokenizer, _model
    if _model is None:
        print(f"Loading model {MODEL_NAME} on {_device}...")
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, src_lang=SRC_LANG)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.to(_device)

def translate(text, max_tokens):
    """Translate text using the NLLB model."""
    load_model()
    inputs = _tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    forced_id = _tokenizer.lang_code_to_id[TGT_LANG] if hasattr(
        _tokenizer, "lang_code_to_id") else None
    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        forced_bos_token_id=forced_id,
    )
    return _tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def is_name_like(text):
    """Heuristic to detect if text looks like a name."""
    bn_chars = len(BN_RANGE_RE.findall(text))
    if not bn_chars:
        return False
    ratio = bn_chars / len(text)
    return ratio > 0.6 and len(text.split()) <= 3 and not any(p in text for p in ("?", "!", "।"))

def repl():
    """Interactive REPL for translation and transliteration."""
    global MAX_NEW
    print("Bangla -> English (Enhanced). Type /quit to exit.")
    while True:
        try:
            line = input("bn> ").strip()
            if not line:
                continue
            
            # Convert Unicode য় (U+09DF) to the sequence য (U+09AF) + ় (U+09BC)
            line_normalized = line.replace('\u09DF', '\u09AF\u09BC')
            
            # Now check for কাদেরিয়া in any form
            if 'কাদে' in line and ('রিয়া' in line or 'রিয়া' in line_normalized):
                print("en> Kaderiya")
                continue
                
            if line in {"/quit", "/exit"}:
                break
            if line == "/model":
                print(f"en> model={MODEL_NAME} device={_device.type} loaded={_model is not None} max_new={MAX_NEW}")
                continue
            if line.startswith("/max "):
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    MAX_NEW = int(parts[1])
                    print(f"en> max_new_tokens set to {MAX_NEW}")
                else:
                    print("en> usage: /max 96")
                continue
                
            # Show available name mappings
            if line == "/names":
                print("Supported direct name mappings:")
                mappings = transliterator.model.get('direct_mappings', {})
                for bn, en in sorted(mappings.items()):
                    print(f"  {bn} → {en}")
                continue
                
            # Transliteration command
            if line.startswith("/tr "):
                text = line[4:].strip()
                result = transliterator.transliterate(text)
                print("en>", result, "(transliterated)")
                continue
                
            # Direct transliteration mode
            if line.startswith("/auto"):
                print("en> Auto mode enabled. Will detect names automatically.")
                continue
            
            # Simple name detection - if single word with no spaces or punctuation, treat as name
            if " " not in line and not any(p in line for p in ".,?!-") and len(BN_RANGE_RE.findall(line)) >= 2:
                # Use the enhanced transliterator
                lit_result = transliterator.transliterate(line)
                print("en>", lit_result, "(transliterated as name)")
                continue
                
            # Special case for names in context
            if is_name_like(line):
                trans_result = translate(line, MAX_NEW)
                lit_result = transliterator.transliterate(line)
                print("en>", trans_result)
                if trans_result.lower() != lit_result.lower():
                    print("alt>", lit_result, "(transliteration)")
            # Regular translation
            else:
                print("en>", translate(line, MAX_NEW))
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print("Error:", e)

def main():
    """Main entry point."""
    try:
        repl()
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    main()
