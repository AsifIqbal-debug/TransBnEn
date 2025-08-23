# TransBnEn

A comprehensive Bangla-English translation and transliteration tool with trained machine learning model.

## Features

- Transliterate Bengali (Bangla) names to English
- Proper handling of special Bengali character combinations 
- Trained model using real Bengali-English name pairs
- Integration with NLLB translation model for complete sentences
- Direct mapping system for common names
- Interactive command-line interface
- Debugging tools for character-by-character analysis
- Title/prefix handling (e.g., "শ্রী" → "Shri", "ডাঃ" → "Dr.")
- Surname/postfix handling (e.g., "চক্রবর্তী" → "Chakraborty")
- Support for names with both prefixes and suffixes

## Installation

```
pip install -e .
```

## Usage

### Command Line

```
# Basic usage
python -m TransBnEn "আসিফ"

# Interactive mode
python -m TransBnEn -i

# Enhanced mode (uses trained model)
python -m TransBnEn -e -i

# Translation mode (requires transformers, torch)
python -m TransBnEn -e -t
```

### As a Library

```python
# Basic transliteration
from TransBnEn.transliterate import transliterate
en_name = transliterate("আসিফ")
print(en_name)  # Output: Asif

# Enhanced transliteration with trained model
from TransBnEn.enhanced_transliterator import EnhancedTransliterator
transliterator = EnhancedTransliterator()
en_name = transliterator.transliterate("আসিফ")
print(en_name)  # Output: Asif
```

### Training Your Own Model

```
# Train with custom data
python train_model.py --data your_data.csv

# Test the trained model
python train_model.py --test
```

### Debugging

```
python scripts/debug_transliteration.py "কাদেরিয়া"
```

## Data Format

The training data should be a CSV file with two columns:
1. Bengali name
2. English transliteration

Example:
```
Bengali,English
আসিফ,Asif
কাদেরিয়া,Kaderiya
```

## Special Name Handling

### Prefix/Title Support
The system properly handles Bengali title prefixes with their conventional English transliterations:

Examples:
- "শ্রী" → "Shri" (Mr.)
- "শ্রীমতী" → "Srimoti" (Mrs.)
- "ডাঃ" → "Dr." (Doctor)
- "প্রফেসর" → "Professor"
- "মোঃ" → "Md." (Mohammad)
- "জনাব" → "Janab" (Mr. in Muslim context)

### Surname/Postfix Support
The system preserves conventional English spellings for Bengali surnames:

Examples:
- "চক্রবর্তী" → "Chakraborty"
- "বন্দ্যোপাধ্যায়" → "Banerjee"
- "মুখোপাধ্যায়" → "Mukherjee"
- "দত্ত" → "Datta"
- "সেন" → "Sen"

### Father Name Transliteration
```python
from TransBnEn.translateIndigo_with_father_name import transliterate_father_name

result = transliterate_father_name("মোঃ আরিফ খান")  # "Md. Arif Khan"
```

### Complex Name Examples
```python
from TransBnEn.translateIndigo_with_father_name import transliterate

# Name with title and surname
result = transliterate("শ্রী অভিজিৎ চক্রবর্তী")  # "Shri Abhijit Chakraborty"

# Multi-word name with title and surname
result = transliterate("শ্রীমতী কল্যাণী দেবী চট্টোপাধ্যায়")  # "Srimoti Kalyani Devi Chatterjee"
```

## Testing

```
python test_surname_postfix.py  # Test surname handling
python test_combined_prefix_suffix.py  # Test names with both prefixes and suffixes
```
python -m unittest discover tests
```
