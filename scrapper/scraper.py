import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def scrape_with_xpath(target_url, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    try:
        driver.get(target_url)
        time.sleep(5)
        
        # This XPath says: "Find every 'a' tag anywhere in the document 
        # that has an 'href' attribute containing the word 'pdffiles'"
        xpath_query = "//a[contains(@href, 'pdffiles')]"
        pdf_links = driver.find_elements(By.XPATH, xpath_query)
        
        print(f"Found {len(pdf_links)} links using XPath.")

        for i, link in enumerate(pdf_links):
            url = link.get_attribute("href")
            # We can even get the text of the link to name our file!
            title = link.text.strip() or f"law_doc_{i}"
            
            print(f"Downloading: {title}")
            res = requests.get(url)
            filename = os.path.join(download_dir, f"{i}_{url.split('/')[-1]}")
            
            with open(filename, 'wb') as f:
                f.write(res.content)
                
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_with_xpath("https://pakistancode.gov.pk/english/UHyrdu.php", "data/raw")