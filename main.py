from helper import color_print
from api_utils.review_utils import fetch_reviews, save_to_csv
from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset, Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)
import transformers
from trl import SFTTrainer
from peft import LoraConfig
import re
import wandb
import warnings

warnings.filterwarnings("ignore")
wandb.init(mode="disabled")
load_dotenv()

token = os.getenv("G2_API_TOKEN")
hf_token = os.getenv("HF_TOKEN")
category_mapping = {
    "Comparative Analysis": "Comparative Analysis",
    "Data Analysis Tools": "Data Analysis Tools",
    "Customer Satisfaction ": "Customer Satisfaction",
    "User Experience": "User Experience",
    "Support Quality": "Support Quality",
    "Missing Functionality": "Missing Functionality",
    "Training": "Training and Documentation",
    "Documentation": "Training and Documentation",
    "Multi-Language Support": "Multi-Language Support",
    "Customer Satisfaction": "Customer Satisfaction",
    "Collaboration Features": "Collaboration Features",
    "Pricing": "Pricing",
    "Integration Capabilities": "Integration Capabilities",
    "Mobile Accessibility": "Mobile Accessibility",
    "Product Benefits": "Product Benefits",
    "Customization Options": "Customization Options",
    '"Product Benefits"': "Product Benefits",
    '"Integration Capabilities"': "Integration Capabilities",
    "User Experience ": "User Experience",
    "Bug Fixes": "Bug Fixes",
    "Ease of Setup": "Ease of Setup",
    "Automation Capabilities": "Automation Capabilities",
    "Product Benefits ": "Product Benefits",
    '"The review doesnt provide enough information to be categorized."': "Other",
    "Application Performance": "Application Performance",
    '"Ease of Setup"': "Ease of Setup",
    '"Customer Satisfaction"': "Customer Satisfaction",
}


color_print("Welcome to our project!", "green")
color_print(
    "This is a simple project to demonstrate the use of few-shot learning to extract categories from given reviews.",
    "green",
)
color_print("We will be using the following models for this purpose.", "green")
color_print("1. Gemma-2B", "blue")
color_print("2. Zephyr-3B", "magenta")
color_print("3. Phi-2", "yellow")


color_print("Fetching reviews from the G2 API", "cyan")
reviews = fetch_reviews(token)
color_print("Fetched reviews successfully", "green")
color_print("Saving reviews to CSV", "cyan")
save_to_csv(reviews, "g2_reviews.csv")
color_print("Reviews saved to 'g2_reviews.csv'", "green")

color_print("Extracting categories from reviews", "cyan")
df = pd.read_csv("data_with_categories (1).csv")
dataset = Dataset.from_pandas(df)


# Apply the updated function to the dataset
def formatting_prompts_func(examples):
    outputs = examples["predicted_categories"]
    texts = []
    for output in outputs:
        output = output.replace("'", "")
        if "\\n" in output:
            output = output.replace("\\n", " and ")
        if "'- " in output:
            output = output.replace("'- ", "").replace("\n", " and ")
        output = output.replace("- ", "").replace("[", "").replace("]", "")
        output = output.split(" and ")
        # Remove numbers from the output
        output = [re.sub(r"\d+", "", item) for item in output]
        # Map the categories using the category_mapping
        output = [category_mapping.get(item.strip(), "Other") for item in output]
        output = ", ".join(output)
        output = output.replace(". ", "")
        texts.append(output)
    return {"cats": texts}


dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=4)
dataset = dataset.remove_columns(
    ["title", "rating", "liked", "disliked", "predicted_categories"]
)

feature_sets = [
    "Application Performance",
    "User Experience",
    "Missing Functionality",
    "Bug Fixes",
    "Customer Satisfaction",
    "Comparative Analysis",
    "Pricing",
    "Ease of Setup",
    "Support Quality",
    "Product Benefits",
    "Security Features",
    "Customization Options",
    "Integration Capabilities",
    "Scalability",
    "Mobile Accessibility",
    "Multi-Language Support",
    "Data Analysis Tools",
    "Collaboration Features",
    "Training and Documentation",
    "Automation Capabilities",
]

color_print("Setting up Gemma-2B model for few-shot learning", "cyan")


lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model_ids = ["google/gemma-2b", "stabilityai/stablelm-zephyr-3b", "microsoft/phi-2"]
model_id = model_ids[0]

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0})
gemmaprompt = """You are an assistant for classifying a product review into feature sets. Classify the input into 2 appropriate feature sets. The list of feature sets available to categorize from are :
'['Application Performance', 'User Experience', 'Missing Functionality', 'Bug Fixes', 'Customer Satisfaction', 'Comparative Analysis', 'Pricing', 'Ease of Setup', 'Support Quality', 'Product Benefits','Security Features','Customization Options','Integration Capabilities','Scalability','Mobile Accessibility','Multi-Language Support','Data Analysis Tools','Collaboration Features','Training and Documentation','Automation Capabilities']'
### Input:
{}

### Response:
{}
"""
zephyrprompt = """
<|user|>
{}
<|assistant|>
{}
"""
EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    instructions = examples["text"]
    inputs = examples["cats"]
    texts = []
    for instruction, input in zip(instructions, inputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = gemmaprompt.format(instruction, input) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset.train_test_split(test_size=0.1)

trainer = SFTTrainer(
    model=model,
    dataset_text_field="text",
    max_seq_length=256,
    train_dataset=dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        max_steps=50,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_32bit",
    ),
    peft_config=lora_config,
)
trainer.train()


color_print("Finished finetuning Gemma-2B model.", "cyan")

prompt_selected = gemmaprompt.format(
    "Best Tech Review Site on the Internet G2 Crowd provides a very impartial and fully verified technology software review forum. The technology sections are well defined and nicely summarized. Overall, it's very professional and you can actually count on the review provided since they can't be manipulated by the vendors. Really like the way that we can embed the review widgets on our own website with links to our optimized landing page. The environment provides a great user experience for visitors to our website and illustrates how open we are to input from our customers. The style of review presentation is also very easy for visitors to consume. It's similar to consumer based review sites, but with much higher data quality. We count on the fact that prospects reading our reviews can trust the customer input. Difficult to say what I dislike, but if I had to choose I would say that sometimes the technology categories are too general. This can result sometimes in very different, non-competitive products in the same category. This makes products hard to compare at times.",
    "",
)
inputs = tokenizer([prompt_selected] * 1, return_tensors="pt").to("cuda")
output_tokens = model.generate(**inputs, max_new_tokens=32)
generated_text_gemma = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

## phi-2
color_print("Setting up Microsoft Phi-2 model for few-shot learning", "cyan")
model_ids = ["AsphyXIA/gemma-g2", "stabilityai/stablelm-zephyr-3b", "microsoft/phi-2"]
model_id = model_ids[2]

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0})


color_print("Finished finetuning Microsoft Phi-2 model.", "cyan")

prompt_selected = "Given a product review: 'Best Tech Review Site on the Internet G2 Crowd provides a very impartial and fully verified technology software review forum. The technology sections are well defined and nicely summarized. Overall, it's very professional and you can actually count on the review provided since they can't be manipulated by the vendors. Really like the way that we can embed the review widgets on our own website with links to our optimized landing page. The environment provides a great user experience for visitors to our website and illustrates how open we are to input from our customers. The style of review presentation is also very easy for visitors to consume. It's similar to consumer based review sites, but with much higher data quality. We count on the fact that prospects reading our reviews can trust the customer input. Difficult to say what I dislike, but if I had to choose I would say that sometimes the technology categories are too general. This can result sometimes in very different, non-competitive products in the same category. This makes products hard to compare at times.' Classify the review into 2 of the most relevant categories it falls under: ['Application Performance', 'User Experience', 'Missing Functionality', 'Bug Fixes', 'Customer Satisfaction', 'Comparative Analysis', 'Pricing', 'Ease of Setup', 'Support Quality', 'Product Benefits','Security Features','Customization Options','Integration Capabilities','Scalability','Mobile Accessibility','Multi-Language Support','Data Analysis Tools','Collaboration Features','Training and Documentation','Automation Capabilities']"
inputs = tokenizer([prompt_selected] * 1, return_tensors="pt").to("cuda")
output_tokens = model.generate(**inputs, max_new_tokens=200)
generated_text_phi2 = tokenizer.decode(output_tokens[0], skip_special_tokens=True)


##zephyr
color_print("Setting up Zephyr model for few-shot learning", "cyan")
model_ids = ["AsphyXIA/gemma-g2", "stabilityai/stablelm-zephyr-3b", "microsoft/phi-2"]
model_id = model_ids[1]
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0})
zephyrprompt = """
<|user|>
{}
<|assistant|>
{}
"""
color_print("Finished finetuning Zephyr model.", "cyan")

prompt_selected = zephyrprompt.format(
    "Best Tech Review Site on the Internet G2 Crowd provides a very impartial and fully verified technology software review forum. The technology sections are well defined and nicely summarized. Overall, it's very professional and you can actually count on the review provided since they can't be manipulated by the vendors. Really like the way that we can embed the review widgets on our own website with links to our optimized landing page. The environment provides a great user experience for visitors to our website and illustrates how open we are to input from our customers. The style of review presentation is also very easy for visitors to consume. It's similar to consumer based review sites, but with much higher data quality. We count on the fact that prospects reading our reviews can trust the customer input. Difficult to say what I dislike, but if I had to choose I would say that sometimes the technology categories are too general. This can result sometimes in very different, non-competitive products in the same category. This makes products hard to compare at times. Classify the above product review into 2 of the most relevant categories it falls under. the list of categories are : ['Application Performance', 'User Experience', 'Missing Functionality', 'Bug Fixes', 'Customer Satisfaction', 'Comparative Analysis', 'Pricing', 'Ease of Setup', 'Support Quality', 'Product Benefits','Security Features','Customization Options','Integration Capabilities','Scalability','Mobile Accessibility','Multi-Language Support','Data Analysis Tools','Collaboration Features','Training and Documentation','Automation Capabilities']",
    "",
)

inputs = tokenizer([prompt_selected] * 1, return_tensors="pt").to("cuda")
output_tokens = model.generate(**inputs, max_new_tokens=200)
generated_text_zephyr = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

gemma_text = generated_text_gemma.split("### Response:")
gemma_text = gemma_text[1].strip()
gemma_categories = gemma_text.split(",")
color_print(f"Google Gemma Categories: {gemma_categories}", "cyan")
phi2_text = generated_text_phi2.split("##OUTPUT")
phi2_text = phi2_text[1]
phi2_categories = []
for item in feature_sets:
    if item in phi2_text:
        phi2_categories.append(item)
color_print(f"Microsoft Phi-2 Categories: {gemma_categories}", "yellow")

zephyr_text = generated_text_zephyr.split("<|assistant|>")
zephyr_text = zephyr_text[1].strip()
zephyr_categories = []
for item in feature_sets:
    if item in zephyr_text:
        zephyr_categories.append(item)
color_print(f"Stability AI Zephyr Categories: {gemma_categories}", "magenta")

set1 = set(gemma_categories)
set2 = set(phi2_categories)
set3 = set(zephyr_categories)

color_print("Combining categories from all models", "cyan")
union_set = set1.union(set2, set3)
union = list(union_set)

color_print(f"Combined Categories: {union}", "green")
