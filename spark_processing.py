from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("G2ReviewsPreprocessing") \
    .getOrCreate()

# Read the CSV file from HDFS
reviews_df = spark.read.option("header", "true").csv("hdfs://127.0.0.1:9000/user/baliga/g2_reviews.csv")

# Display the schema of the dataframe
reviews_df.printSchema()

# Remove irrelevant characters, correct misspellings, and handle missing values
reviews_cleaned_df = reviews_df.select(
    col("id"),
    lower(regexp_replace("attributes", "[^a-zA-Z0-9\\s]", "")).alias("cleaned_attributes")
)

# Tokenization
tokenizer = Tokenizer(inputCol="cleaned_attributes", outputCol="tokens")
reviews_tokenized_df = tokenizer.transform(reviews_cleaned_df)

# Remove stopwords
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
reviews_filtered_df = remover.transform(reviews_tokenized_df)

# Select relevant fields
selected_fields_df = reviews_filtered_df.select(
    col("id"),
    col("filtered_tokens")
)

# Display the preprocessed data
selected_fields_df.show(truncate=False)

# Stop SparkSession
spark.stop()
