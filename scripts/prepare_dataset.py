from google.cloud import translate_v2 as translate
import requests
import pandas as pd
import trafilatura
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHROMEDRIVER_PATH = "/path/to/chromedriver"  # Update to your Chromedriver path

def fetch_url_content(url: str) -> Optional[str]:
    """
    Fetch the content of a URL using requests or Selenium if necessary.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        Optional[str]: The main text content of the page or None if failed.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)

        if response.status_code == 200:
            downloaded_content = response.content
        else:
            logging.warning(f"Request failed for {url}. Falling back to Selenium.")

            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

            service = Service(CHROMEDRIVER_PATH)
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get(url)

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//body'))
                )
                page_source = driver.page_source
                downloaded_content = page_source.encode('utf-8')
            finally:
                driver.quit()

        content = trafilatura.bare_extraction(downloaded_content, with_metadata=False)
        return content.raw_text or None

    except Exception as e:
        logging.error(f"Failed to fetch content from {url}: {e}")
        return None

def translate_text(text: str, target_language: str = 'en') -> Optional[str]:
    """
    Translate text into the specified target language using Google Translate API.

    Args:
        text (str): The text to translate.
        target_language (str): The language code to translate the text into.

    Returns:
        Optional[str]: The translated text or None if failed.
    """
    try:
        translate_client = translate.Client()
        translation = translate_client.translate(text, target_language=target_language)
        return translation.get('translatedText')
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return None

def process_urls(urls: List[str], target_language: str = 'en') -> Dict[str, Optional[str]]:
    """
    Process a list of URLs to fetch and translate their content.

    Args:
        urls (List[str]): List of URLs to process.
        target_language (str): Language code for translation (default is 'en').

    Returns:
        Dict[str, Optional[str]]: A dictionary with URLs as keys and their translated content as values.
    """
    results = {}
    for url in tqdm(urls, desc="Processing URLs"):
        content = fetch_url_content(url)
        if content:
            translated_content = translate_text(content, target_language)
            results[url] = translated_content
        else:
            results[url] = None
    return results

def process_csv_file(file_path: str, target_language: str = 'en') -> None:
    """
    Read a CSV file, fetch translations for URLs, and save the results back into the file.

    Args:
        file_path (str): Path to the CSV file.
        target_language (str): Language code for translation.
    """
    try:
        df = pd.read_csv(file_path)

        if 'url' not in df.columns:
            logging.error(f"No 'url' column found in {file_path}.")
            return

        urls = df['url'].dropna().tolist()
        logging.info(f"Processing {len(urls)} URLs from {file_path}.")

        translations = process_urls(urls, target_language)

        df['Translation'] = df['url'].map(translations)
        df.to_csv(file_path, index=False)

        logging.info(f"Updated file saved as {file_path}.")

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    file_paths = [
        "./data/gt/ground_truth_extraction.csv",
        "./data/gt/ground_truth_filtering.csv",
        "./data/raw/fact_checking_articles.csv"
    ]

    for file_path in file_paths:
        process_csv_file(file_path)