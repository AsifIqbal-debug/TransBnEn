#!/usr/bin/env python3
"""Tests for the transliteration functions."""

import unittest
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TransBnEn.transliterate import transliterate, NAME_MAPPINGS

class TransliterationTest(unittest.TestCase):
    """Test transliteration functions."""

    def test_name_mappings(self):
        """Test direct name mappings."""
        for bn_name, en_name in NAME_MAPPINGS.items():
            self.assertEqual(transliterate(bn_name), en_name)
    
    def test_specific_names(self):
        """Test specific name transliterations."""
        test_cases = [
            ("আসিফ", "Asif"),
            ("কাদেরিয়া", "Kaderiya"),
            ("রহিম", "Rahim"),
            ("আমিনা", "Amina"),
            ("জমিলা", "Jamila"),
            ("ফয়সাল", "Faysal"),
        ]
        
        for bn_name, expected in test_cases:
            self.assertEqual(transliterate(bn_name), expected)
    
    def test_endings(self):
        """Test correct handling of common Bengali endings."""
        test_cases = [
            ("আসিয়া", "Asiya"),
            ("শোকরিয়া", "Shokriya"),
            ("রাজিয়া", "Rajiya"),
        ]
        
        for bn_name, expected in test_cases:
            self.assertEqual(transliterate(bn_name), expected)

if __name__ == "__main__":
    unittest.main()
