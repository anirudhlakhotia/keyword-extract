# keyword-extract
# Abnormal Distribution: Extracting Product Categories from G2 Reviews Using LLM

Submission for G2 Hack

Our project analyzes customer reviews and extract key feature sets that customers are looking for in products or services. <br/> We use a pre-trained language model to extract keywords from the reviews based on few-shot learning and then use clustering algorithms to group the keywords into categories. We then use these categories to identify the key features that customers are looking for in the products or services.

## Algorithm Overview, How are reviews suggested to customers?

1. **Data Collection**: Retrieve review data from the G2 API.

2. **Preprocessing**: Clean and preprocess the review text to remove duplicates, empty rows, and irrelevant information.

3. **Model Feeding**:
   - **Gemini Model**: Feed preprocessed review text into the Gemini language model.
   - **Zephyr Model**: Feed preprocessed review text into the Zephyr language model.
   - **Phi2 Model**: Feed preprocessed review text into the Phi2 language model.

4. **Combining Predictions**:
    - Collect predictions from Gemini, Zephyr, and Phi2 models.
    - Combine the predictions from all models to create a unified set of categories.
    - Assign categories to input review text based on the combined predictions.

## Architecture Diagram
![Architecture](https://github.com/anirudhlakhotia/keyword-extract/assets/52605103/85774fb3-79dd-48eb-ba48-8721a73de559)

## Using Small Language Models for Enhanced Efficiency

In our project, we leverage the capabilities of small language models to enhance efficiency in various aspects of our workflow. These smaller models, such as **stabilityai/stablelm-zephyr-3b** or **microsoft/phi-2**, play a crucial role in several key areas:.

1. **Fine-tuning and Transfer Learning**: We leverage small language models for fine-tuning and transfer learning tasks. By starting with a pre-trained small language model and fine-tuning it on our specific domain or task, we can achieve better performance and adaptability.

2. **Efficient Resource Utilization**: We optimize resource utilization in our project. These models require less computational power and memory compared to larger models, allowing us to process larger datasets and perform computations more efficiently.

## Commands to Run the Project




## Useful Links



## Team Members
- Anirudh Lakhotia
- Akash Kamalesh
- Anshul Baliga




