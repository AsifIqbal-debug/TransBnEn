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

## Testing

```
python -m unittest discover tests
```
