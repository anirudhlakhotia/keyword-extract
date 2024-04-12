import csv
import json

def extract_question_data(json_data):
    extracted_data = []
    for question in json_data['data']:
        question_id = question['id']
        title = question['attributes']['title']
        text_template = question['attributes']['text_template']
        is_required = question['attributes']['is_required']
        question_type = question['attributes']['question_type']
        input_type = question['attributes']['input_type']
        
        extracted_data.append({
            'question_id': question_id,
            'title': title,
            'text_template': text_template,
            'is_required': is_required,
            'question_type': question_type,
            'input_type': input_type
        })
    
    return extracted_data

def write_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question_id', 'title', 'text_template', 'is_required', 'question_type', 'input_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

with open('new4.json', 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)

extracted_data = extract_question_data(json_data)

# Write extracted data to CSV file
write_to_csv(extracted_data, 'extracted_questions.csv')
