import requests
import json
from bs4 import BeautifulSoup
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# environment variables
load_dotenv()
CONNECTION_STRING = os.getenv('CONNECTION_STRING')

# MongoDB
client = MongoClient(CONNECTION_STRING)
db = client["APTs"]
collection = db["blackberry_vendor"]

# all the MITRE ATT&CK tactics
TACTICS = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
    "Network Effects",
    "Network Service Effects",
    "Inhibit Response Function",
    "Impair Process Control"
]

def extract_text_from_url(url: str) -> str:
    """
    Extract the concatenated text of all <p> tags from a given URL.

    Parameters:
    url (str): The URL of the webpage to extract text from.

    Returns:
    str: The concatenated text of all <p> tags.
    """
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to load page with status code {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    
    corpus = ' '.join([p.get_text() for p in paragraphs])
    
    return corpus


def find_and_remove_tactics(text: str) -> (str, list):
    """
    Find and remove tactics that appear in the given text.

    Parameters:
    text (str): The text to search for tactics.

    Returns:
    (str, list): The modified text with tactics removed, and a list of tactics found in the text.
    """
    found_tactics = []
    for tactic in TACTICS:
        if tactic in text:
            found_tactics.append(tactic)
            text = text.replace(tactic, '')
    return text, found_tactics

def process_url(url: str):
    """
    Process the given URL to extract the text and find tactics, then insert into MongoDB.

    Parameters:
    url (str): The URL of the webpage to process.
    """
    try:
        corpus = extract_text_from_url(url)
        corpus, tactics = find_and_remove_tactics(corpus)
        
        document = {
            "url": url,
            "corpus": corpus,
            "tactics": tactics
        }
        
        collection.insert_one(document)
        print(f"Inserted document for URL: {url}")
    except Exception as e:
        print(f"An error occurred while processing URL: {url}. Error: {str(e)}")


def extract_all_reports():
    """
    Extract and process all reports from the paginated blog URL until there are no more pages.

    """
    page_num = 1
    while True:             # unfortunately we have to loop indifinitely until we reach the last page
        url = f"https://blogs.blackberry.com/bin/blogs?page={page_num}&category=https://blogs.blackberry.com/en/category/research-and-intelligence&locale=en"
        response = requests.get(url)
        try:
            if response.status_code == 200:
                data = json.loads(response.text)
                if len(data) == 0:
                    break
                else:
                    print(f"[+] Processing page {page_num}")
                    for blog_post in data:
                        post_url = blog_post['url']
                        process_url(post_url)
                    page_num += 1
            else:
                print(f"[-] Failed to retrieve data from page {page_num} with status code {response.status_code}")
                break
        except Exception as e:
            print(f"[-] An unexpected error has occurred. Check the URL in the browser to see if anything has changed. {url}")
            print(str(e))
            break



if __name__ == "__main__":
    extract_all_reports()
