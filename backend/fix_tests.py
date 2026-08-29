import os

def replace_in_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

# Fix test_api_pipeline.py
replace_in_file('tests/test_api_pipeline.py', {
    '"email_address": "test@komute.com",': '"email_address": "test@komute.com",\n            "role": "Driver",',
    'year=2020,': 'year="2020",'
})

# Fix test_dl_ocr.py
replace_in_file('tests/test_dl_ocr.py', {
    'date_of_birth="2000-01-01",': 'date_of_birth="2000-01-01",\n        email_address="test@test.com",\n        mobile_number="1234567890",\n        role="Driver",'
})

# Fix test_vehicle_alpr.py
replace_in_file('tests/test_vehicle_alpr.py', {
    'year=2020,': 'year="2020",'
})

# Fix benchmark.py
replace_in_file('benchmark.py', {
    'email_address="test@test.com",': 'email_address="test@test.com",\n        role="Driver",',
    'year=2020,': 'year="2020",'
})
