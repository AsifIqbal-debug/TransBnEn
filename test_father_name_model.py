from TransBnEn.translateIndigo_with_father_name import load_father_name_model, transliterate_father_name

print('Loading father name model...')
model = load_father_name_model()
print(f'Model loaded with {len(model["direct_mappings"])} mappings')

# Unicode encoded Bengali names
test_names = [
    'মোঃ মধু মিয়া',
    'বিনোদ চন্দ্র',
    'দেবেন চন্দ্র দাস',
    'শ্রী প্রশন্ন চন্দ্র দাস'
]

print('\nTesting father name transliteration:')
for name in test_names:
    result = transliterate_father_name(name)
    print(f'{name} -> {result}')
