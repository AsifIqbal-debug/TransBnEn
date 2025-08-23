"""Script to test variations of the Bengali name in the translator."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from TransBnEn.translateIndigo_with_father_name import transliterate_father_name

def test_variations():
    test_variations = [
        'শ্রী অম্বী চন্দ্র সরকার',
        "'শ্রী অম্বী চন্দ্র সরকার",
        "শ্রী অম্বী চন্দ্র সরকার'",
        ' শ্রী অম্বী চন্দ্র সরকার ',
        '  শ্রী  অম্বী  চন্দ্র  সরকার  '
    ]

    print('Testing variations of the Bengali name:')
    for i, variant in enumerate(test_variations, 1):
        result = transliterate_father_name(variant)
        print(f'{i}. Input: "{variant}"')
        print(f'   Output: "{result}"')

if __name__ == "__main__":
    test_variations()
