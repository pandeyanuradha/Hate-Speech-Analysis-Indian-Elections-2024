"""
CrowdTangle Data Fetcher
========================
This script downloads posts (and optionally images) from CrowdTangle for Facebook or Instagram.
It supports:
- Paginated requests
- Batch JSON saving
- Optional image downloads
- Rate-limiting to avoid API throttling
"""

import json
import os
import sys
import getopt
import time
from urllib import request, error
from datetime import datetime, timedelta
from ratelimit import limits
import requests
import traceback
import logging

# ----------------------
# Logging configuration
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ----------------------
# Configuration
# ----------------------
DOWNLOAD_IMAGES = True  # Set to False if you don't want to download images
API_CALLS_PER_MINUTE = 5

# ----------------------
# Helper Functions
# ----------------------
@limits(calls=API_CALLS_PER_MINUTE, period=60, raise_on_limit=False)
def api_call(request_url):
    """Perform a GET request with rate limiting"""
    return requests.get(request_url).json()


def execute_call(request_url):
    """Execute API call with retry logic on failure"""
    logging.info(f"Requesting URL: {request_url}")
    while True:
        try:
            return api_call(request_url)
        except Exception:
            logging.error("Failed to obtain or parse JSON response. Retrying in 10 minutes...")
            traceback.print_exc()
            time.sleep(10 * 60)


def retrieve_images_for_result(json_result, platform):
    """Save media (images) from posts if present"""
    for post_item in json_result.get('posts', []):
        if 'media' not in post_item:
            continue

        i = 0
        for media_item in post_item['media']:
            try:
                if media_item['type'] == 'photo':
                    account_name = "".join([c for c in post_item['account']['name']
                                            if c.isalnum() or c == ' ']).rstrip()
                    account_id = post_item['account']['platformId']
                    folder = os.path.join(platform, f"{account_name}-{account_id}")
                    os.makedirs(folder, exist_ok=True)

                    link = media_item['url']
                    filename = os.path.join(folder, "".join([c for c in post_item['date']
                                                             if c.isalnum() or c == ' ']).rstrip() + f' {i}')

                    # Save post JSON
                    with open(filename + ".json", "w", encoding="utf-8") as f:
                        json.dump(post_item, f, ensure_ascii=False, indent=4)

                    # Optionally download image
                    if DOWNLOAD_IMAGES and link:
                        if not os.path.exists(filename + ".jpg"):
                            try:
                                request.urlretrieve(link, filename + ".jpg")
                                logging.info(f"Image saved: {filename}.jpg")
                            except error.HTTPError:
                                logging.warning(f"Failed to download image: {filename}.jpg | {link}")
                    i += 1
            except Exception:
                logging.error('Failed to store media for post.')
                traceback.print_exc()


def paginate_request(platform, url, request_name):
    """Handle paginated API responses"""
    api_response = execute_call(url)

    while api_response.get('status') == 200:
        os.makedirs(platform, exist_ok=True)

        # Save API JSON response
        with open(os.path.join(platform, request_name + ".json"), "w", encoding="utf-8") as f:
            json.dump(api_response["result"], f, ensure_ascii=False, indent=4)
        logging.info(f"API response saved: {request_name}.json")

        # Save images
        retrieve_images_for_result(api_response["result"], platform)

        next_page = api_response["result"].get('pagination', {}).get('nextPage')
        if not next_page:
            break
        api_response = execute_call(next_page)

    if api_response.get('status') != 200:
        logging.error(f"Error: {api_response.get('status')} - See CrowdTangle API docs for help.")


# ----------------------
# Main Execution
# ----------------------
def main():
    """
    Usage:
        python crowdtangle_fetch.py -f -o <parameters> -t <days> -s <start-date> --token <CROWD_TOKEN>
    """

    argv = sys.argv[1:]
    platform = 'facebook'
    token = os.getenv("CROWDTANGLE_TOKEN", "")
    parameters = ''
    time_interval = ''
    start_date = ''

    # Parse CLI arguments
    try:
        opts, args = getopt.getopt(argv, "hfio:t:s:", ["options=", "time-interval=", "startdate=", "token="])
    except getopt.GetoptError:
        logging.error('Usage: python crowdtangle_fetch.py -f/-i -o <parameters> -t <time-interval in days> -s <start-date> --token <token>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logging.info('Usage: python crowdtangle_fetch.py -f/-i -o <parameters> -t <time-interval in days> -s <start-date> --token <token>')
            sys.exit()
        elif opt == "-f":
            platform = 'facebook'
        elif opt == "-i":
            platform = 'instagram'
        elif opt in ('-o', '--options'):
            parameters = arg
        elif opt in ('-t', '--time-interval'):
            time_interval = arg
        elif opt in ('-s', '--start-date'):
            start_date = arg
        elif opt == '--token':
            token = arg

    if not token:
        logging.error("CrowdTangle API token not provided! Use --token or set CROWDTANGLE_TOKEN env variable.")
        sys.exit(1)

    base_path = "https://api.crowdtangle.com/posts"

    # Option 1: Start-date based fetching
    if start_date:
        end = datetime.now()
        start = datetime.strptime(start_date, '%Y-%m-%d')
        i = 0
        while start < end:
            request_name = f"{platform}-request-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}-{i}"
            url = (f"{base_path}?token={token}&sortBy=date&count=100&{parameters}"
                   f"&startDate={start.strftime('%Y-%m-%dT%H:%M:%S')}"
                   f"&endDate={end.strftime('%Y-%m-%dT%H:%M:%S')}")
            paginate_request(platform, url, request_name)
            end = start
            start = start - timedelta(days=1)
            i += 1

    # Option 2: Time-interval based continuous fetching
    if time_interval:
        time_interval = int(time_interval)
        last_execution_date = datetime.now() - timedelta(days=time_interval)

        while True:
            current_execution_date = datetime.now() - timedelta(hours=1)
            request_name = f"{platform}-request-{current_execution_date.strftime('%Y-%m-%dT%H:%M:%S')}"
            url = (f"{base_path}?token={token}&sortBy=date&count=100&{parameters}"
                   f"&startDate={last_execution_date.strftime('%Y-%m-%dT%H:%M:%S')}"
                   f"&endDate={current_execution_date.strftime('%Y-%m-%dT%H:%M:%S')}")
            paginate_request(platform, url, request_name)

            last_execution_date = current_execution_date
            logging.info("Waiting until next timed request...")
            time.sleep(86_400 * time_interval)


if __name__ == "__main__":
    main()
