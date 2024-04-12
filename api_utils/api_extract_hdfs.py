import csv
import requests
import json
from hdfs import InsecureClient
from dotenv import load_dotenv
import os

load_dotenv()


def fetch_reviews(token):
    headers = {"Authorization": f"Bearer {token}"}
    reviews = []
    page = 1
    while True:
        params = {"page[size]": 100, "page[number]": page}
        response = requests.get(
            "https://data.g2.com/api/v1/survey-responses",
            headers=headers,
            params=params,
        )
        if response.status_code == 200:
            batch_reviews = response.json().get("data", [])
            if not batch_reviews:
                break
            reviews.extend(batch_reviews)
            page += 1
        else:
            print(f"Failed to fetch reviews with status code: {response.status_code}")
            break
    return reviews


def extract_info(attributes):
    # Parse attributes string as JSON
    attributes_json = json.loads(attributes)
    # Extract relevant information
    product_name = attributes_json.get("product_name", "")
    star_rating = attributes_json.get("star_rating", "")
    title = attributes_json.get("title", "")
    # Extract love and hate values from comment_answers
    love_value = (
        attributes_json.get("comment_answers", {}).get("love", {}).get("value", "")
    )
    hate_value = (
        attributes_json.get("comment_answers", {}).get("hate", {}).get("value", "")
    )
    return product_name, star_rating, title, love_value, hate_value


def save_to_csv(reviews, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = reviews[0].keys() if reviews else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow(review)


def read_csv_and_extract(filename):
    reviews = {}
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            id = row.get("id", "")
            attributes = row.get("attributes", "")
            # Extract relevant information from attributes
            product_name, star_rating, title, love_value, hate_value = extract_info(
                attributes
            )
            # Store extracted information in a dictionary for each ID
            reviews[id] = {
                "product_name": product_name,
                "star_rating": star_rating,
                "title": title,
                "love_value": love_value,
                "hate_value": hate_value,
            }
    return reviews


def write_extracted_csv(reviews, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "id",
            "product_name",
            "star_rating",
            "title",
            "love_value",
            "hate_value",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for id, info in reviews.items():
            info["id"] = id
            writer.writerow(info)


if __name__ == "__main__":
    token = os.getenv("G2_API_KEY")

    reviews = fetch_reviews(token)

    save_to_csv(reviews, "g2_reviews.csv")
    print("Reviews fetched and saved to 'g2_reviews.csv'")

    # Initialize HDFS client
    client = InsecureClient("http://127.0.0.1:9000", user="baliga")

    # Write reviews data to HDFS
    with client.write("/user/baliga/g2_reviews.csv", encoding="utf-8") as writer:
        with open("g2_reviews.csv", "rb") as f:
            writer.write(f)

    extracted_reviews = read_csv_and_extract("g2_reviews.csv")

    write_extracted_csv(extracted_reviews, "extracted_reviews.csv")
