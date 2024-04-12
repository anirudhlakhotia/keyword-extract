# keyword-extract
# Abnormal Distribution: Extracting Product Categories from G2 Reviews Using LLM

Submission for G2 Hack

Our project analyzes customer reviews and extract key feature sets that customers are looking for in products or services. <br/> We use a pre-trained language model to extract keywords from the reviews based on few-shot learning and then use clustering algorithms to group the keywords into categories. We then use these categories to identify the key features that customers are looking for in the products or services.

## Features

- **Data Collection**: Fetches review data from the G2 API in batches of 100 using the provided endpoint.
- **Preprocessing**: Cleans and preprocesses the review text by removing duplicates, empty rows, and irrelevant information.
- **Few-Shot Learning**: Utilizes few-shot learning with a pre-trained large language model (LLM) to identify and categorize product features based on provided prompts.
- **Category-based Analysis**: Optionally categorizes reviews into predefined categories and performs analysis within each category.
- **Voting Classifier**: Use predictions of Gemma 2B, Zephyr 3B, and Microsoft Phi 2 models, fine-tuned on review data, and assign categories to input review text based on the majority vote among these models

## Architecture Diagram
![Architecture](https://github.com/anirudhlakhotia/keyword-extract/assets/52605103/85774fb3-79dd-48eb-ba48-8721a73de559)
