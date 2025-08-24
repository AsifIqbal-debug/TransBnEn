from TransBnEn.translateIndigo_with_father_name import load_father_name_model, transliterate_father_name

print('Loading father name model...')
model = load_father_name_model()
print(f'Model loaded with {len(model["direct_mappings"])} mappings')

# Test the specific name
test_name = 'মোঃ খুশি মিয়া'
result = transliterate_father_name(test_name)
print(f'\nTest result: {test_name} -> {result}')
