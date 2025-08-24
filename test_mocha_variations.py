"""Script to test variations of 'মোছাঃ' in the translator."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from TransBnEn.translateIndigo_with_name import transliterate_name_specialized

def test_variations():
    """Test variations of 'মোছাঃ' in the translator."""
    test_variations = [
        'মোছাঃ',
        'মোছাঃ ছামিনা বেগম',
        'মোছাঃ  ছামিনা  বেগম',
        "'মোছাঃ ছামিনা বেগম",
        "মোছাঃ ছামিনা বেগম'",
        "মোছাঃ তহমিনা বেগম"
    ]

    print('Testing variations of মোছাঃ:')
    for i, variant in enumerate(test_variations, 1):
        result = transliterate_name_specialized(variant)
        print(f'{i}. Input: "{variant}"')
        print(f'   Output: "{result}"')

if __name__ == "__main__":
    test_variations()
