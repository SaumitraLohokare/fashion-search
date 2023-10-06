import json
import requests
from bs4 import BeautifulSoup
from pprint import pprint
from tqdm import tqdm

session = requests.Session()

URL = 'https://newyork.craigslist.org/search/cla#search=1~gallery~{}~0'

def get_page(start: int = 0, end: int = 71):
    for i in range(start, end):
        yield URL.format(i)

def extract_image_tags(url):
    image_tags = []  # List to store image tags
    
    # Send a request to the URL and get the page content
    response = requests.get(url)
    with open('test.html', 'w') as f:
        f.write(str(response.content))
    
    # Check if the request was successful
    if response.status_code != 200:
        print('Failed to fetch the webpage.')
        return image_tags
    
    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the script tag with the specified id
    script_tag = soup.find('script', {'id': 'ld_searchpage_results'})
    
    if script_tag:
        # Extract the content from the script tag
        ld_json_content = script_tag.string
        
        # Parse the JSON content
        ld_json = json.loads(ld_json_content)
    
    return ld_json

pages = get_page()

csv_file = open('items.csv', 'w')

csv_file.write('Name,Description,Images...\n')

for page in tqdm(pages):
    for item in extract_image_tags(page)['itemListElement']:
        if item['@type'] == 'ListItem':
            description = item['item']['description'] if 'description' in item['item'].keys() else ''
            if 'image' not in item['item'].keys():
                tqdm.write('Skipping because image not there')
                continue

            images = item['item']['image']
            if len(images) == 0:
                tqdm.write('Skipping because images are 0')
                continue

            name = item['item']['name']
            images_list = ','.join([img for img in images])
            csv_file.write(f'"{name}","{description}",{images_list}\n')
    
csv_file.close()