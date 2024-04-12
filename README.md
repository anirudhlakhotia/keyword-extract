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

## Using Small Language Models for Enhanced Efficiency

In our project, we leverage the capabilities of small language models to enhance efficiency in various aspects of our workflow. These smaller models, such as stabilityai/stablelm-zephyr-3b or microsoft/phi-2, play a crucial role in several key areas:

1. Automated Dataset Cleaning: We utilize small language models to develop a pipeline for automated dataset cleaning. By leveraging these models, we streamline the data cleaning process, improving dataset quality and reliability.

2. Enhanced Reasoning Ability: Our project incorporates small language models to enhance the reasoning ability of the final model. By integrating techniques for logical reasoning and inference using datasets like meta-math/MetaMathQA or microsoft/orca-math-word-problems-200k, we augment the model's capacity for complex reasoning tasks.

3. Token Counting and Distribution Analysis: Small language models enable us to quantify the number of tokens within any given dataset for any given tokenizer. This functionality serves as a fundamental tool for analyzing token distributions and comprehending vocabulary dimensions across datasets, facilitating more informed decision-making in our project workflow.

4. Efficient Resource Utilization: By utilizing small language models, we optimize resource utilization in our project. These models require less computational power and memory compared to larger models, allowing us to process larger datasets and perform computations more efficiently.

5. Fine-tuning and Transfer Learning: We leverage small language models for fine-tuning and transfer learning tasks. By starting with a pre-trained small language model and fine-tuning it on our specific domain or task, we can achieve better performance and adaptability.

## Algorithm Overview, How are reviews suggested to customers?

1. **Data Collection**: Retrieve review data from the G2 API.

2. **Preprocessing**: Clean and preprocess the review text to remove duplicates, empty rows, and irrelevant information.

3. **Model Feeding**:
   - **Gemini Model**: Feed preprocessed review text into the Gemini language model.
   - **Zephyr Model**: Feed preprocessed review text into the Zephyr language model.
   - **Phi2 Model**: Feed preprocessed review text into the Phi2 language model.

4. **Voting Mechanism**:
   - Collect predictions from Gemini, Zephyr, and Phi2 models.
   - Implement a voting mechanism to determine the most common categories among the models' predictions.
   - Assign categories to input review text based on the majority vote among the models.

5. **Target Product Identification**: Determine target product categories based on the output of the voting mechanism.

