import re
category_mapping = {
    'Comparative Analysis': 'Comparative Analysis',
    'Data Analysis Tools': 'Data Analysis Tools',
    'Customer Satisfaction ': 'Customer Satisfaction',
    'User Experience': 'User Experience',
    'Support Quality': 'Support Quality',
    'Missing Functionality': 'Missing Functionality',
    'Training': 'Training and Documentation',
    'Documentation': 'Training and Documentation',
    'Multi-Language Support': 'Multi-Language Support',
    'Customer Satisfaction': 'Customer Satisfaction',
    'Collaboration Features': 'Collaboration Features',
    'Pricing': 'Pricing',
    'Integration Capabilities': 'Integration Capabilities',
    'Mobile Accessibility': 'Mobile Accessibility',
    'Product Benefits': 'Product Benefits',
    'Customization Options': 'Customization Options',
    '"Product Benefits"': 'Product Benefits',
    '"Integration Capabilities"': 'Integration Capabilities',
    'User Experience ': 'User Experience',
    'Bug Fixes': 'Bug Fixes',
    'Ease of Setup': 'Ease of Setup',
    'Automation Capabilities': 'Automation Capabilities',
    'Product Benefits ': 'Product Benefits',
    '"The review doesnt provide enough information to be categorized."': 'Other',
    'Application Performance': 'Application Performance',
    '"Ease of Setup"': 'Ease of Setup',
    '"Customer Satisfaction"': 'Customer Satisfaction',
    # Add any other mappings as needed
}

def formatting_prompts_func(examples):
    outputs = examples["predicted_categories"]
    texts = []
    for output in outputs:
        output = output.replace("'","")
        if "\\n" in output:
            output = output.replace("\\n"," and ")
        if "'- " in output:
            output = output.replace("'- ", "").replace("\n"," and ")
        output = output.replace("- ","").replace("[", "").replace("]", "")
        output = output.split(" and ")
        # Remove numbers from the output
        output = [re.sub(r'\d+', '', item) for item in output]
        # Map the categories using the category_mapping
        output = [category_mapping.get(item.strip(), 'Other') for item in output]
        output = ', '.join(output)
        output = output.replace(". ","")
        texts.append(output)
    return {"cats": texts}

