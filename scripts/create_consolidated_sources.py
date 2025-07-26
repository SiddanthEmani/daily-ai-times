#!/usr/bin/env python3
"""
Create consolidated sources.yaml file

This script reads all existing source YAML files and combines them
with new sources from the test results into a single sources.yaml file.
"""

import yaml
from pathlib import Path

def load_existing_sources():
    """Load all existing sources from YAML files"""
    # Get the script directory and navigate to sources
    script_dir = Path(__file__).parent
    sources_dir = script_dir.parent / "src" / "shared" / "config" / "sources"
    all_sources = {}
    
    # Load from each YAML file
    yaml_files = ["government.yaml", "industry.yaml", "media.yaml", "open_source.yaml", "research.yaml"]
    
    for yaml_file in yaml_files:
        file_path = sources_dir / yaml_file
        if file_path.exists():
            print(f"Loading sources from {yaml_file}...")
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'sources' in data:
                    for source_name, source_info in data['sources'].items():
                        # Only keep name, url, description
                        clean_source = {
                            'url': source_info.get('url'),
                            'description': source_info.get('description', '')
                        }
                        all_sources[source_name] = clean_source
    
    return all_sources

def add_new_sources():
    """Add the 79 new sources from test results"""
    new_sources = {
        # AI/Technology sources
        'mit_tech_review': {
            'url': 'https://www.technologyreview.com/feed',
            'description': 'MIT Technology Review'
        },
        'mit_tech_review_ai': {
            'url': 'https://www.technologyreview.com/topic/artificial-intelligence/feed',
            'description': 'Artificial intelligence – MIT Technology Review'
        },
        'the_verge': {
            'url': 'https://www.theverge.com/rss/index.xml',
            'description': 'The Verge - Technology news and media'
        },
        'the_verge_ai': {
            'url': 'https://www.theverge.com/rss/ai-artificial-intelligence/index.xml',
            'description': 'AI | The Verge'
        },
        'ars_technica': {
            'url': 'https://feeds.arstechnica.com/arstechnica/index',
            'description': 'Ars Technica - All content'
        },
        'ars_technica_ai': {
            'url': 'https://arstechnica.com/ai/feed',
            'description': 'AI – Ars Technica'
        },
        'techcrunch': {
            'url': 'https://techcrunch.com/feed',
            'description': 'TechCrunch - Startup and Technology News'
        },
        'techcrunch_ai': {
            'url': 'https://techcrunch.com/category/artificial-intelligence/feed',
            'description': 'AI News & Artificial Intelligence | TechCrunch'
        },
        'wired': {
            'url': 'https://www.wired.com/feed/rss',
            'description': 'WIRED - Technology news'
        },
        'ieee_spectrum': {
            'url': 'https://spectrum.ieee.org/feeds/feed.xml',
            'description': 'IEEE Spectrum - Technology news'
        },
        'ieee_spectrum_ai': {
            'url': 'https://spectrum.ieee.org/feeds/topic/artificial-intelligence.xml',
            'description': 'IEEE Spectrum - Artificial Intelligence'
        },
        'venturebeat': {
            'url': 'https://venturebeat.com/feed',
            'description': 'VentureBeat - Transformative tech coverage'
        },
        'venturebeat_ai': {
            'url': 'https://venturebeat.com/category/ai/feed',
            'description': 'AI News | VentureBeat'
        },
        'cnet_news': {
            'url': 'https://www.cnet.com/rss/news',
            'description': 'CNET - Technology news'
        },
        'zdnet_news': {
            'url': 'https://www.zdnet.com/news/rss.xml',
            'description': 'ZDNet - Latest news'
        },
        'zdnet_ai': {
            'url': 'https://www.zdnet.com/topic/artificial-intelligence/rss.xml',
            'description': 'ZDNet - Artificial Intelligence'
        },
        'engadget': {
            'url': 'https://www.engadget.com/rss.xml',
            'description': 'Engadget - Consumer electronics and gadgets'
        },
        'anandtech': {
            'url': 'https://www.anandtech.com/rss',
            'description': 'AnandTech - Computer hardware'
        },
        'toms_hardware': {
            'url': 'https://www.tomshardware.com/feeds/all',
            'description': 'Tom\'s Hardware - Computer hardware news'
        },
        'the_register': {
            'url': 'https://www.theregister.com/headlines.atom',
            'description': 'The Register - Enterprise Technology News'
        },
        'openai_news': {
            'url': 'https://openai.com/news/rss.xml',
            'description': 'OpenAI News'
        },
        'deepmind_blog': {
            'url': 'https://deepmind.google/blog/rss.xml',
            'description': 'Google DeepMind Blog'
        },
        'mit_ai_news': {
            'url': 'https://news.mit.edu/rss/topic/artificial-intelligence2',
            'description': 'MIT News - Artificial intelligence'
        },
        'google_research': {
            'url': 'https://research.google/blog/rss',
            'description': 'The latest research from Google'
        },
        'microsoft_research': {
            'url': 'https://www.microsoft.com/en-us/research/feed',
            'description': 'Microsoft Research'
        },
        'nvidia_blog': {
            'url': 'https://blogs.nvidia.com/feed',
            'description': 'NVIDIA Blog'
        },
        'stanford_ai_blog': {
            'url': 'https://ai.stanford.edu/blog/feed.xml',
            'description': 'The Stanford AI Lab Blog'
        },
        'berkeley_ai_blog': {
            'url': 'https://bair.berkeley.edu/blog/feed.xml',
            'description': 'The Berkeley Artificial Intelligence Research Blog'
        },
        'hacker_news': {
            'url': 'https://news.ycombinator.com/rss',
            'description': 'Hacker News - Links for the intellectually curious'
        },
        'ai_news': {
            'url': 'https://www.artificialintelligence-news.com/feed',
            'description': 'AI News - Artificial Intelligence News'
        },
        'machine_learning_mastery': {
            'url': 'https://machinelearningmastery.com/blog/feed',
            'description': 'MachineLearningMastery.com - Making developers awesome at machine learning'
        },
        'towards_data_science': {
            'url': 'https://towardsdatascience.com/feed',
            'description': 'Towards Data Science - AI, ML & data-science insights'
        },
        
        # Entertainment sources
        'variety': {
            'url': 'https://variety.com/feed/',
            'description': 'Variety - Entertainment news and film reviews'
        },
        'variety_film': {
            'url': 'http://feeds.feedburner.com/variety/news/film',
            'description': 'Variety - Film News'
        },
        'variety_tv': {
            'url': 'http://feeds.feedburner.com/variety/news/tv',
            'description': 'Variety - TV News'
        },
        'variety_music': {
            'url': 'http://feeds.feedburner.com/variety/news/music',
            'description': 'Variety - Music News'
        },
        'hollywood_reporter': {
            'url': 'https://hollywoodreporter.com/feed/',
            'description': 'The Hollywood Reporter - Movie and TV news'
        },
        'deadline': {
            'url': 'https://deadline.com/feed/',
            'description': 'Deadline - Hollywood Entertainment Breaking News'
        },
        'indiewire': {
            'url': 'https://indiewire.com/feed/',
            'description': 'IndieWire - The Voice of Creative Independence'
        },
        'screen_rant': {
            'url': 'https://screenrant.com/feed/',
            'description': 'Screen Rant - Entertainment news and reviews'
        },
        'collider': {
            'url': 'https://collider.com/feed/',
            'description': 'Collider - Movie news and reviews'
        },
        'rolling_stone_music': {
            'url': 'https://rollingstone.com/music/feed/',
            'description': 'Music – Rolling Stone'
        },
        'billboard': {
            'url': 'https://billboard.com/feed/',
            'description': 'Billboard - Music Charts and News'
        },
        'pitchfork': {
            'url': 'https://pitchfork.com/feed/rss/',
            'description': 'Pitchfork - Music reviews and news'
        },
        'nme': {
            'url': 'https://nme.com/feed/',
            'description': 'NME - Music and pop culture news'
        },
        'polygon': {
            'url': 'https://polygon.com/rss/index.xml',
            'description': 'Polygon - Video games, sci-fi, fantasy, and entertainment'
        },
        'tmz': {
            'url': 'https://tmz.com/rss.xml',
            'description': 'TMZ.com - Celebrity Gossip and Entertainment News'
        },
        'e_online': {
            'url': 'https://eol-feeds.eonline.com/rssfeed/us/top_stories',
            'description': 'E! Online (US) - Top Stories'
        },
        'slashfilm': {
            'url': 'https://feeds2.feedburner.com/slashfilm',
            'description': '/Film - Movie News and Reviews'
        },
        'comingsoon': {
            'url': 'https://comingsoon.net/feed/',
            'description': 'ComingSoon.net – Movie Trailers and News'
        },
        'firstshowing': {
            'url': 'https://firstshowing.net/feed/',
            'description': 'FirstShowing.net - Connecting Hollywood with its Audience'
        },
        
        # Sports sources
        'espn_top': {
            'url': 'https://www.espn.com/espn/rss/news',
            'description': 'ESPN - Top Sports News'
        },
        'espn_nfl': {
            'url': 'https://www.espn.com/espn/rss/nfl/news',
            'description': 'ESPN - NFL News'
        },
        'espn_nba': {
            'url': 'https://www.espn.com/espn/rss/nba/news',
            'description': 'ESPN - NBA News'
        },
        'espn_mlb': {
            'url': 'https://www.espn.com/espn/rss/mlb/news',
            'description': 'ESPN - MLB News'
        },
        'espn_nhl': {
            'url': 'https://www.espn.com/espn/rss/nhl/news',
            'description': 'ESPN - NHL News'
        },
        'espn_soccer': {
            'url': 'https://www.espn.com/espn/rss/soccer/news',
            'description': 'ESPN - Soccer News'
        },
        'yahoo_sports': {
            'url': 'https://sports.yahoo.com/rss',
            'description': 'Yahoo! Sports - News, Scores, Standings'
        },
        'yahoo_sports_nfl': {
            'url': 'https://sports.yahoo.com/nfl/rss',
            'description': 'Yahoo Sports - NFL News'
        },
        'yahoo_sports_nba': {
            'url': 'https://sports.yahoo.com/nba/rss',
            'description': 'Yahoo Sports - NBA News'
        },
        'yahoo_sports_mlb': {
            'url': 'https://sports.yahoo.com/mlb/rss',
            'description': 'Yahoo Sports - MLB News'
        },
        'bbc_sport': {
            'url': 'https://feeds.bbci.co.uk/sport/rss.xml',
            'description': 'BBC Sport - Sports news'
        },
        'bbc_sport_football': {
            'url': 'https://feeds.bbci.co.uk/sport/football/rss.xml',
            'description': 'BBC Sport - Football'
        },
        'bbc_sport_cricket': {
            'url': 'https://feeds.bbci.co.uk/sport/cricket/rss.xml',
            'description': 'BBC Sport - Cricket'
        },
        'bbc_sport_tennis': {
            'url': 'https://feeds.bbci.co.uk/sport/tennis/rss.xml',
            'description': 'BBC Sport - Tennis'
        },
        'sky_sports': {
            'url': 'https://www.skysports.com/rss/12040',
            'description': 'SkySports - Sports News'
        },
        'nbc_sports': {
            'url': 'https://www.nbcsports.com/index.atom',
            'description': 'NBC Sports - Sports news and scores'
        },
        'deadspin': {
            'url': 'https://deadspin.com/rss',
            'description': 'Deadspin - Sports News Without Fear, Favor or Compromise'
        },
        'mlb_news': {
            'url': 'https://www.mlb.com/feeds/news/rss.xml',
            'description': 'MLB News'
        },
        'yankees_news': {
            'url': 'https://www.mlb.com/yankees/feeds/news/rss.xml',
            'description': 'Yankees News'
        },
        'hockey_writers': {
            'url': 'https://thehockeywriters.com/feed/',
            'description': 'The Hockey Writers - NHL News and Analysis'
        },
        'mlb_trade_rumors': {
            'url': 'https://www.mlbtraderumors.com/feed',
            'description': 'MLB Trade Rumors'
        },
        'guardian_sport': {
            'url': 'https://www.theguardian.com/sport/rss',
            'description': 'Sport | The Guardian'
        },
        'guardian_football': {
            'url': 'https://www.theguardian.com/football/rss',
            'description': 'Football | The Guardian'
        },
        'sportskeeda': {
            'url': 'https://www.sportskeeda.com/feed',
            'description': 'Sportskeeda - Latest Sports News'
        },
        
        # Health sources
        'cdc_travel_notices': {
            'url': 'https://wwwnc.cdc.gov/travel/rss/notices.xml',
            'description': 'Travel Notices - CDC Travelers\' Health'
        },
        'cdc_newsroom': {
            'url': 'https://tools.cdc.gov/api/v2/resources/media/132608.rss',
            'description': 'CDC Online Newsroom'
        },
        'medicinenet_daily': {
            'url': 'https://www.medicinenet.com/rss/dailyhealth.xml',
            'description': 'MedicineNet Daily News'
        },
        'medical_xpress': {
            'url': 'https://medicalxpress.com/rss-feed',
            'description': 'Medical Xpress - latest medical and health news stories'
        }
    }
    
    return new_sources

def create_consolidated_sources():
    """Create consolidated sources.yaml file"""
    print("Creating consolidated sources.yaml file...")
    
    # Load existing sources
    existing_sources = load_existing_sources()
    print(f"Loaded {len(existing_sources)} existing sources")
    
    # Add new sources
    new_sources = add_new_sources()
    print(f"Adding {len(new_sources)} new sources")
    
    # Combine all sources
    all_sources = {**existing_sources, **new_sources}
    
    # Create the YAML structure
    sources_data = {
        'sources': all_sources
    }
    
    # Write to consolidated file
    script_dir = Path(__file__).parent
    output_path = script_dir.parent / "src" / "shared" / "config" / "sources.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(sources_data, f, default_flow_style=False, sort_keys=True)
    
    print(f"Created consolidated sources.yaml with {len(all_sources)} total sources")
    print(f"Breakdown: {len(existing_sources)} existing + {len(new_sources)} new = {len(all_sources)} total")

if __name__ == "__main__":
    create_consolidated_sources() 