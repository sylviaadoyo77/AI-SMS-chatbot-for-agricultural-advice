# add_articles.py
"""import pymysql
from dotenv import load_dotenv
import os
import uuid
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

def get_db():
    """Establishes and returns a new database connection."""
    try:
        conn = pymysql.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return None

def add_sample_articles():
    """
    Adds a predefined set of agricultural knowledge articles to the database.
    This script is idempotent; it checks for existing articles before inserting.
    """
    articles = [
        # Crop Management
        {
            "title": "Mahindi - Upandaji Bora",
            "content": "Panda mahindi wakati wa mvua za masika. Tumia mbolea ya NPK kwa kiwango cha kg 50 kwa ekari moja. Umbali wa kupanda: cm 75 kati ya mstari, cm 30 kati ya mimea.",
            "keywords": "mahindi,upandaji,mbolea,upandaji bora",
            "category": "crop_management"
        },
        {
            "title": "Viazi - Uvunaji na Uhifadhi",
            "content": "Vuna viazi pale majani yanapoanza kukauka. Weka kwenye sehemu baridi na yenye ukingo. Epuka kuyaweka moja kwa moja juu ya sakafu.",
            "keywords": "viazi,uvunaji,uhifadhi,mazao",
            "category": "harvest_storage"
        },
        
        # Pest Control
        {
            "title": "Kudhibiti Mba za Mahindi",
            "content": "Tumia dawa ya imidakloprid kwa kuzipulizia mimea. Pima lita 20 za maji na chanjisha dawa ya ml 10. Rudia kila wiki 2.",
            "keywords": "mba,mahindi,dawa,wadudu",
            "category": "pest_control"
        },
        {
            "title": "Kudhibiti Uvundo Nyekundu wa Maharage",
            "content": "Panda mbegu zilizokwamishwa na dawa ya fungicide. Epuka kupanda katika maeneo yenye unyevu mwingi. Zungusha mazao.",
            "keywords": "uvundo,maharage,fungicide,magonjwa",
            "category": "disease_control"
        },
        
        # Soil Health
        {
            "title": "Kuboresha Udongo Dobi",
            "content": "Ongeza mbolea ya kienyeji na maganda ya mimea. Pima pH ya udongo na rekebisha kwa kutumia chokaa kwa udongo wenye asidi.",
            "keywords": "udongo,dobi,mbolea,kienyeji",
            "category": "soil_health"
        },
        
        # Weather Patterns
        {
            "title": "Kilimo Kipya cha Mvua",
            "content": "Anza kupanda siku 2 baada ya mvua kubwa ya kwanza. Tumia mbegu za mimea inayokomaa haraka kuepuka ukame.",
            "keywords": "mvua,kilimo,ukame,mimea",
            "category": "weather_patterns"
        }
    ]
    
    db = get_db()
    if not db:
        return
        
    try:
        with db.cursor() as cursor:
            for article in articles:
                # Check if article with the same title already exists
                cursor.execute(
                    "SELECT article_id FROM KNOWLEDGE_ARTICLE WHERE title = %s",
                    (article['title'],)
                )
                if cursor.fetchone():
                    logging.info(f"Article '{article['title']}' already exists, skipping.")
                    continue
                
                # Insert new article if it doesn't exist
                cursor.execute(
                    "INSERT INTO KNOWLEDGE_ARTICLE (article_id, title, content, keywords, category) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (str(uuid.uuid4()), article['title'], article['content'], article['keywords'], article['category'])
                )
                logging.info(f"Added: {article['title']}")
                
        db.commit()
        logging.info("Knowledge base updated successfully!")
        
    except Exception as e:
        logging.error(f"Error adding articles: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == '__main__':
    add_sample_articles()

