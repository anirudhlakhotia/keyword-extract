import csv
import requests

def fetch_reviews(token):
    headers = {
        'Authorization': f'Bearer {token}'
    }
    reviews = []
    page = 1
    while True:
        # Fetch reviews in batches of 100
        params = {'page[size]': 100, 'page[number]': page}
        response = requests.get('https://data.g2.com/api/v1/survey-responses', headers=headers, params=params)
        if response.status_code == 200:
            batch_reviews = response.json().get('data', [])
            if not batch_reviews:
                break
            reviews.extend(batch_reviews)
            page += 1
        else:
            print(f"Failed to fetch reviews with status code: {response.status_code}")
            break
    return reviews

def save_to_csv(reviews, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = reviews[0].keys() if reviews else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow(review)

if __name__ == "__main__":
    token = '64d916344e608810a8ce23f37a554165efe5a6474b1929572187d0b8c9af33ef'
    reviews = fetch_reviews(token)
    save_to_csv(reviews, 'new3.csv')
    print("Reviews fetched and saved to 'g2_reviewss.csv'")
