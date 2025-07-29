"""
Data Collection for SEAL-drip
Simple, clean implementation for collecting daily data from:
- Wikipedia recent changes
- HackerNews front page 
- arXiv RSS feed
"""

import requests
import feedparser
import json
import jsonlines
from datetime import datetime, timedelta
from typing import List, Dict, Any
from tqdm import tqdm
import time
import config

def collect_hackernews() -> List[Dict[str, Any]]:
    """
    Collect top stories from HackerNews.
    Returns list of story dictionaries with: title, url, score, time, text
    """
    print("📰 Collecting HackerNews stories...")
    
    try:
        # Get top story IDs
        response = requests.get(config.HACKERNEWS_TOP_URL, timeout=10)
        response.raise_for_status()
        story_ids = response.json()[:config.HACKERNEWS_MAX_ITEMS]
        
        stories = []
        for story_id in tqdm(story_ids, desc="Fetching HN stories"):
            try:
                story_response = requests.get(
                    config.HACKERNEWS_ITEM_URL.format(story_id), 
                    timeout=5
                )
                story_response.raise_for_status()
                story_data = story_response.json()
                
                if story_data and story_data.get('type') == 'story':
                    stories.append({
                        'id': story_data.get('id'),
                        'title': story_data.get('title', ''),
                        'url': story_data.get('url', ''),
                        'score': story_data.get('score', 0),
                        'time': story_data.get('time', 0),
                        'text': story_data.get('text', ''),
                        'by': story_data.get('by', ''),
                        'descendants': story_data.get('descendants', 0),
                        'collected_at': datetime.now().isoformat()
                    })
                
                # Be nice to HN's API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching story {story_id}: {e}")
                continue
        
        print(f"✅ Collected {len(stories)} HackerNews stories")
        return stories
        
    except Exception as e:
        print(f"❌ Error collecting HackerNews: {e}")
        return []

def collect_arxiv() -> List[Dict[str, Any]]:
    """
    Collect recent papers from arXiv CS RSS feed.
    Returns list of paper dictionaries with: title, summary, authors, link
    """
    print("📚 Collecting arXiv papers...")
    
    try:
        feed = feedparser.parse(config.ARXIV_RSS_URL)
        
        papers = []
        for entry in feed.entries[:config.ARXIV_MAX_ITEMS]:
            papers.append({
                'title': entry.get('title', '').strip(),
                'summary': entry.get('summary', '').strip(),
                'authors': [author.get('name', '') for author in entry.get('authors', [])],
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'tags': [tag.get('term', '') for tag in entry.get('tags', [])],
                'collected_at': datetime.now().isoformat()
            })
        
        print(f"✅ Collected {len(papers)} arXiv papers")
        return papers
        
    except Exception as e:
        print(f"❌ Error collecting arXiv: {e}")
        return []

def collect_wikipedia() -> List[Dict[str, Any]]:
    """
    Collect recent Wikipedia changes.
    For now, this is a simplified version - we'll use the featured feed as a proxy.
    In production, you'd want to use the actual recent changes API.
    """
    print("🌐 Collecting Wikipedia changes...")
    
    try:
        # This is simplified - normally you'd use the recent changes API
        # For now, let's collect from today's featured content as a demo
        today = datetime.now()
        url = f"https://en.wikipedia.org/api/rest_v1/feed/featured/{today.year}/{today.month:02d}/{today.day:02d}"
        
        response = requests.get(url, timeout=10)
        
        # If today fails, try yesterday
        if response.status_code == 404:
            yesterday = today - timedelta(days=1)
            url = f"https://en.wikipedia.org/api/rest_v1/feed/featured/{yesterday.year}/{yesterday.month:02d}/{yesterday.day:02d}"
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            changes = []
            
            # Process featured article if exists
            if 'tfa' in data:
                tfa = data['tfa']
                changes.append({
                    'type': 'featured_article',
                    'title': tfa.get('title', ''),
                    'extract': tfa.get('extract', ''),
                    'content_urls': tfa.get('content_urls', {}),
                    'collected_at': datetime.now().isoformat()
                })
            
            # Process featured image if exists  
            if 'image' in data:
                image = data['image']
                changes.append({
                    'type': 'featured_image',
                    'title': image.get('title', ''),
                    'description': image.get('description', {}).get('text', ''),
                    'collected_at': datetime.now().isoformat()
                })
            
            print(f"✅ Collected {len(changes)} Wikipedia items")
            return changes
        else:
            print(f"❌ Wikipedia API returned status {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error collecting Wikipedia: {e}")
        return []

def save_data(data: List[Dict[str, Any]], source: str) -> str:
    """
    Save collected data to JSONL file.
    Returns the filename where data was saved.
    """
    config.ensure_directories()
    
    filename = config.get_daily_filename(source)
    filepath = f"{config.RAW_DATA_DIR}/{filename}"
    
    with jsonlines.open(filepath, mode='w') as writer:
        for item in data:
            writer.write(item)
    
    print(f"💾 Saved {len(data)} items to {filepath}")
    return filepath

def main():
    """
    Main collection function - runs all collectors and saves data.
    """
    print(f"🚀 Starting SEAL-drip data collection at {datetime.now()}")
    print(f"📁 Data will be saved to: {config.RAW_DATA_DIR}")
    
    # Collect from all sources
    collectors = [
        ('hackernews', collect_hackernews),
        ('arxiv', collect_arxiv),
        ('wikipedia', collect_wikipedia)
    ]
    
    results = {}
    for source_name, collector_func in collectors:
        print(f"\n--- Collecting from {source_name} ---")
        data = collector_func()
        if data:
            filepath = save_data(data, source_name)
            results[source_name] = {
                'count': len(data),
                'filepath': filepath
            }
        else:
            results[source_name] = {'count': 0, 'filepath': None}
    
    # Summary
    print(f"\n🎉 Collection complete!")
    total_items = sum(r['count'] for r in results.values())
    print(f"📊 Total items collected: {total_items}")
    
    for source, result in results.items():
        print(f"  - {source}: {result['count']} items")
    
    return results

if __name__ == "__main__":
    main() 