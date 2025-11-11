from apify_client import ApifyClient
import os
from dotenv import load_dotenv
import csv
from datetime import datetime
import ast


load_dotenv()
api_token = os.getenv("APIFY_API_TOKEN")
                      
print(f"Using Apify API Token: {api_token}")

# Initialize the ApifyClient with your API token
client = ApifyClient(api_token)

search_terms = ["حزين", "مكتئب", "طفشان", "مالي خلق", "تعبان نفسيًا", "نفسيتي زفت", "مو قادر أتحمل", "خلاص تعبت", "مو بخير", "ودي أبكي", "قلبي مكسور", "ما فيني طاقة", "أحس بضياع", "تعبت من كل شي", "ما عاد أقدر", "أحس بالحزن", "كل شي يوجع", "فقدت الأمل", "ما لي نفس", ]

# search_query = " OR ".join(search_terms)
# print(f"Search Query: {search_query}")

# Prepare the Actor input
run_input = {
    # "twitterContent": search_query,
    "search_terms": search_terms,
    "maxItems": 200,
    "queryType": "Top",
    "lang": "ar",
    "filter:blue_verified": False,
    "filter:replies": False,
    "filter:media": False,
    "filter:twimg": False,
    "filter:images": False,
    "filter:videos": False,
    "filter:pro_video": False,
    "filter:spaces": False,
    "filter:links": False,
    "filter:mentions": False,
    "filter:news": False,
}

# Run the Actor and wait for it to finish
run = client.actor("CJdippxWmn9uRfooo").call(run_input=run_input)

# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"x_scraper_results_{timestamp}.csv"

def flatten_item(item):
    """Flatten the item by unpacking the author profile object"""
    flattened = {}
    
    for key, value in item.items():
        if key == 'author' and isinstance(value, (dict, str)):
            # If author is a string representation of dict, parse it
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except:
                    # If parsing fails, keep as is
                    flattened[key] = value
                    continue
            
            # Unpack author fields with 'author_' prefix
            if isinstance(value, dict):
                for author_key, author_value in value.items():
                    # Convert nested dicts/lists to string for CSV
                    if isinstance(author_value, (dict, list)):
                        flattened[f'author_{author_key}'] = str(author_value)
                    else:
                        flattened[f'author_{author_key}'] = author_value
            else:
                flattened[key] = value
        else:
            # Convert dicts and lists to strings for CSV compatibility
            if isinstance(value, (dict, list)):
                flattened[key] = str(value)
            else:
                flattened[key] = value
    
    return flattened

# Fetch and save Actor results to CSV as they come
item_count = 0
csv_file = None
writer = None

try:
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        # Flatten the item (unpack author profile)
        flattened_item = flatten_item(item)
        
        # Open file and write header on first item
        if csv_file is None:
            csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
            fieldnames = sorted(flattened_item.keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            print(f"Started saving results to {csv_filename}")
        
        # Write item to CSV
        writer.writerow(flattened_item)
        item_count += 1
        
        # Flush to disk periodically (every 10 items)
        if item_count % 10 == 0:
            csv_file.flush()
            print(f"Saved {item_count} items so far...")
    
    if csv_file:
        csv_file.close()
        print(f"Successfully saved {item_count} items to {csv_filename}")
    else:
        print("No results to save.")
        
except Exception as e:
    if csv_file:
        csv_file.close()
    print(f"Error occurred: {e}")
    raise