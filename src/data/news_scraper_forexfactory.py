import pandas as pd
import time
import random
import os
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def scrape_forexfactory_monthly_archive(start_year=2021):
    """
    Scrapes historical news headlines from ForexFactory using the Monthly Archive URLs.
    Format: https://www.forexfactory.com/news/archive/2024/1
    """
    output_dir = "data/raw_sentiment"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/forexfactory_2021_present.csv"

    # 1. Setup Selenium Stealth
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })

    all_news = []
    seen_ids = set()
    
    # Generate list of months to scrape
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    scrape_tasks = []
    for year in range(start_year, current_year + 1):
        end_month = 12 if year < current_year else current_month
        for month in range(1, end_month + 1):
            scrape_tasks.append((year, month))

    # Reverse to get newest first
    scrape_tasks.reverse()

    print(f"--- Starting Monthly Archive Scrape ({len(scrape_tasks)} months) ---")

    try:
        for year, month in scrape_tasks:
            url = f"https://www.forexfactory.com/news/archive/{year}/{month}"
            print(f"Scraping: {year}-{month:02d}...", end="\r")

            driver.get(url)
            time.sleep(random.uniform(2.5, 4.0))
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            # In Archive pages, news items might be in different blocks or just the news-block__item
            news_items = soup.find_all("div", class_="news-block__item")
            
            if not news_items:
                # Some months might not have "all breaking news" at that URL, or format changed
                # Try fallback: checking for data-items JSON
                print(f"\n[Warning] No news items found for {year}-{month:02d}. Checking fallback...")
                continue

            new_in_month = 0
            for item in news_items:
                try:
                    headline_tag = item.find("div", class_="news-block__title")
                    if not headline_tag: continue
                    a_tag = headline_tag.find("a")
                    if not a_tag: continue
                    
                    title_attr = a_tag.get("title")
                    headline = str(title_attr).strip() if title_attr else a_tag.get_text().strip()
                    
                    href_attr = a_tag.get("href")
                    item_url = str(href_attr) if href_attr else ""
                    item_id = item_url.split('/news/')[1].split('-')[0] if '/news/' in item_url else headline
                    
                    if item_id in seen_ids: continue
                    seen_ids.add(item_id)

                    # Extract Timestamp
                    details_tag = item.find("div", class_="news-block__details")
                    date_str, time_str = f"{year}-{month:02d}-01", "00:00:00"
                    if details_tag:
                        time_span = details_tag.find("span", title=True)
                        if time_span:
                            time_attr = time_span.get("title")
                            try:
                                dt = datetime.strptime(str(time_attr), "%b %d, %Y %I:%M%p")
                                date_str = dt.strftime("%Y-%m-%d")
                                time_str = dt.strftime("%H:%M:%S")
                            except:
                                date_str = str(time_attr)
                        else:
                            nowrap_span = details_tag.find("span", class_="nowrap")
                            if nowrap_span:
                                text = nowrap_span.get_text().strip()
                                # Handle cases like "Apr 25, 2026"
                                try:
                                    dt = datetime.strptime(text, "%b %d, %Y")
                                    date_str = dt.strftime("%Y-%m-%d")
                                except:
                                    date_str = text

                    # Extract Impact
                    impact = "Low"
                    impact_img = details_tag.find("img", class_=lambda x: bool(x and "impact-ff" in x)) if details_tag else None
                    if impact_img:
                        classes = impact_img.get("class")
                        if isinstance(classes, list):
                            for c in classes:
                                if "high" in str(c): impact = "High"
                                elif "medium" in str(c): impact = "Medium"
                    
                    news_item = {
                        "date": date_str,
                        "time": time_str,
                        "impact": impact,
                        "headline": headline
                    }
                    all_news.append(news_item)
                    new_in_month += 1
                except:
                    continue

            print(f"Month {year}-{month:02d} complete. New items: {new_in_month}. Total: {len(all_news)}")
            
            # Save progress after every month
            if all_news:
                pd.DataFrame(all_news).to_csv(output_file, index=False)

    finally:
        driver.quit()

    if all_news:
        print(f"\n--- SCRAPE COMPLETE ---")
        print(f"Total News Items: {len(all_news)}")
        print(f"Saved to: {output_file}")
    else:
        print("\n--- SCRAPE FAILED ---")

if __name__ == "__main__":
    # Scrape from 2021 to present
    scrape_forexfactory_monthly_archive(2021)
