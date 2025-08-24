# Bengali-English Name Transliteration Model Renaming

This update renames the "father_name" model to "name" model throughout the codebase to more accurately reflect its purpose.

## Changes Made

1. **File Renaming**:
   - Renamed `translateIndigo_with_father_name.py` to `translateIndigo_with_name.py`
   - Renamed model file from `father_name_transliteration_model_corrected.pkl` to `name_transliteration_model_corrected.pkl`

2. **Code Updates**:
   - Renamed function `is_father_name_like()` to `is_specialized_name_like()`
   - Renamed function `transliterate_father_name()` to `transliterate_name_specialized()`
   - Renamed function `load_father_name_model()` to `load_name_model()`
   - Updated constant from `FATHER_NAME_MODEL_PATH` to `NAME_MODEL_PATH`
   - Updated command line option from `/fn` to `/nm`
   - Updated output messages for better clarity

3. **Test Updates**:
   - Updated all test files to use the new function names
   - Updated import statements to use the new module name
   - Updated references to the renamed functions in test outputs

4. **Backup**:
   - Created backup of original model file in `models_backup/` directory

## Motivation

The original naming as "father name" was inaccurate as the model actually handles specialized name transliteration for both male and female names, including:

- Full names with titles/honorifics
- Names with prefixes and suffixes
- Names with specialized surname components
- Names needing specific transliteration rules

The new naming better represents the true purpose of the model as a general specialized name transliteration system rather than one specifically for father names.

## Testing

All existing tests have been updated to use the new function names and continue to pass as expected. The model functionality remains identical, only the naming has been updated for clarity and accuracy.
