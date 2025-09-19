# nlp_processor.py (with Swahili support)
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import mysql.connector
import os
from dotenv import load_dotenv
import logging
import json
from googletrans import Translator

# Simple NLTK setup - just try to use the data, don't download it
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    # If data is missing, just print a message but don't try to download
    print("NLTK data not found. Some features might not work properly.")

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'but'}
            
        self.lemmatizer = WordNetLemmatizer()
        self.conn = None
        self.translator = Translator()
        
        # Swahili stop words
        self.swahili_stop_words = {
            'ya', 'na', 'wa', 'kwa', 'ni', 'za', 'katika', 'la', 'kuwa', 'kama',
            'hiyo', 'hii', 'ile', 'yao', 'yake', 'hili', 'hivi', 'hvyo', 'huu',
            'yangu', 'yetu', 'yenu', 'zao', 'wake', 'zangu', 'mimi', 'sisi', 'wewe',
            'nyinyi', 'yeye', 'wao', 'huyo', 'hawa', 'hiki', 'hichi', 'hivi', 'hivyo',
            'vyo', 'vya', 'kila', 'kwenye', 'mwa', 'pia', 'lakini', 'au', 'ama',
            'ila', 'budi', 'hata', 'tena', 'mwaka', 'mwezi', 'siku', 'juma', 'muhimu',
            'hasa', 'bila', 'kati', 'hadi', 'baada', 'kabla', 'watu', 'kitu', 'sababu'
        }
        
        # Enhanced keyword dictionaries for Kenyan agriculture
        self.intent_keywords = {
            'pest': ['pest', 'insect', 'bug', 'worm', 'aphid', 'larvae', 'weevil', 'mite', 'nematode', 'locust', 'treat', 'control', 'manage', 'solution', 'get rid of'],
            'disease': ['disease', 'sick', 'infected', 'rot', 'fungus', 'mold', 'blight', 'wilt', 'spot', 'mildew'],
            'soil': ['soil', 'dirt', 'ground', 'fertilizer', 'compost', 'nutrient', 'ph', 'acidic', 'alkaline', 'manure'],
            'weather': ['weather', 'rain', 'sun', 'temperature', 'forecast', 'climate', 'drought', 'dry', 'wet', 'humidity'],
            'crop': ['crop', 'plant', 'harvest', 'yield', 'growth', 'germination', 'seed', 'seedling', 'transplant'],
            'water': ['water', 'irrigation', 'drought', 'moisture', 'hydrate', 'rainfall', 'watering'],
            'market': ['price', 'market', 'sell', 'buy', 'cost', 'profit', 'income', 'value', 'wholesale', 'retail']
        }
        
        # Swahili intent keywords
        self.swahili_intent_keywords = {
            'pest': ['wadudu', 'maduudu', 'funza', 'vimelea', 'mdudu', 'dhibiti', 'tibu', 'suluhisho', 'ondoa'],
            'disease': ['ugonjwa', 'maradhi', 'shida', 'kuharibu'],
            'soil': ['udongo', 'ardhi', 'mchanga', 'rutuba'],
            'weather': ['hali ya hewa', 'mvua', 'jua', 'ukame', 'baridi', 'joto'],
            'crop': ['mazao', 'mmea', 'mbegu', 'mavuno', 'kulima'],
            'water': ['maji', 'umwagiliaji', 'mvua', 'kinyesi'],
            'market': ['soko', 'bei', 'uuzaji', 'mnunuzi', 'gharama']
        }
        
        # Common Kenyan crops and their variants (English and Swahili)
        self.crop_entities = {
            'maize': ['maize', 'corn', 'mealies', 'mahindi'],
            'beans': ['beans', 'bean', 'ndengu', 'green gram', 'maharagwe'],
            'wheat': ['wheat', 'ngano'],
            'rice': ['rice', 'mchele'],
            'potato': ['potato', 'potatoes', 'viazi'],
            'tomato': ['tomato', 'tomatoes', 'nyanya'],
            'cabbage': ['cabbage', 'cabbages', 'cabage', 'kabeji'],
            'sugarcane': ['sugarcane', 'cane', 'sugar cane', 'muwa'],
            'coffee': ['coffee', 'kahawa'],
            'tea': ['tea', 'chai']
        }
        
        # Common pests in Kenyan agriculture (English and Swahili)
        self.pest_entities = {
            'aphids': ['aphid', 'aphids', 'black aphid', 'green aphid', 'vibura', 'pests'],
            'fall_armyworm': ['armyworm', 'fall armyworm', 'caterpillar', 'kiwavi'],
            'stalk_borer': ['borer', 'stalk borer', 'stem borer', 'kiko'],
            'leaf_miner': ['leaf miner', 'miner', 'mchimbaji'],
            'whitefly': ['whitefly', 'white fly', 'nziweupe'],
            'thrips': ['thrips', 'thrip', 'thiripsi'],
            'mites': ['mite', 'mites', 'spider mite', 'chawa'],
            'nematodes': ['nematode', 'nematodes', 'eelworm', 'minyoo']
        }
        
        # Common symptoms and issues (English and Swahili)
        self.symptom_entities = {
            'yellowing': ['yellow', 'yellowing', 'yellow leaves', 'manjano'],
            'wilting': ['wilt', 'wilting', 'drooping', 'kunyauka'],
            'spots': ['spot', 'spots', 'black spot', 'brown spot', 'doa'],
            'holes': ['hole', 'holes', 'eaten', 'chewed', 'shimo', 'kumezwa'],
            'rot': ['rot', 'rotting', 'rotten', 'decay', 'kuoza'],
            'stunted': ['stunted', 'small', 'not growing', 'kukua polepole'],
            'discoloration': ['discolored', 'discoloration', 'color change', 'kubadilisha rangi']
        }
        
    def get_db_connection(self):
        """Get database connection"""
        if self.conn is None or not self.conn.is_connected():
            self.conn = mysql.connector.connect(
                host=os.getenv('DB_HOST'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                database=os.getenv('DB_NAME')
            )
        return self.conn
    
    def detect_language(self, text):
        try:
            # Common Swahili words that are unlikely to appear in English queries
            swahili_words = {
                'ya', 'na', 'kwa', 'ni', 'za', 'katika', 'la', 'kuwa', 'kama',
                'hiyo', 'hii', 'ile', 'yao', 'yake', 'hili', 'hivi', 'hvyo', 'huu',
                'yangu', 'yetu', 'yenu', 'wake', 'mimi', 'sisi', 'wewe', 'nyinyi',
                'yeye', 'wao', 'huyo', 'hawa', 'hiki', 'hichi', 'hivi', 'hivyo',
                'kila', 'kwenye', 'mwa', 'pia', 'lakini', 'au', 'ama', 'ila',
                'hata', 'tena', 'mwaka', 'mwezi', 'siku', 'juma', 'hasa', 'bila',
                'kati', 'hadi', 'baada', 'kabala', 'watu', 'kitu', 'sababu', 'jina',
                'jinsi', 'pamoja', 'hapa', 'pale', 'huko', 'kule', 'hivyo', 'vivyo'
            }
        
            # Common agricultural terms in Swahili
            swahili_agricultural_terms = {
                'kilimo', 'mazao', 'mmea', 'mbegu', 'mavuno', 'kulima', 'shamba',
                'udongo', 'mbolea', 'umwagiliaji', 'mvua', 'jua', 'ukame', 'wadudu',
                'ugonjwa', 'dawa', 'vimelea', 'vibura', 'funza', 'kiwavi', 'kiko',
                'nziweupe', 'thiripsi', 'chawa', 'minyoo', 'mahindi', 'maharagwe',
                'ngano', 'mchele', 'viazi', 'nyanya', 'kabeji', 'muwa', 'kahawa',
                'chai', 'majani', 'mizizi', 'maua', 'matunda', 'maganda', 'machungwa'
            }
        
            text_lower = text.lower()
            words = set(text_lower.split())
        
            # Count Swahili words
            swahili_count = len(words.intersection(swahili_words))
        
            # Count Swahili agricultural terms
            agricultural_count = len(words.intersection(swahili_agricultural_terms))
            # Debug logging
            logger.info(f"Language detection - Text: {text}")
            logger.info(f"Language detection - Swahili words found: {words.intersection(swahili_words)}")
            logger.info(f"Language detection - Agricultural terms found: {words.intersection(swahili_agricultural_terms)}")
            logger.info(f"Language detection - Swahili count: {swahili_count}, Agricultural count: {agricultural_count}")
        
            # If we have a significant number of Swahili words or agricultural terms, it's Swahili
            if (swahili_count > len(words) * 0.2) or (agricultural_count > 0):
                return 'sw'
            else:
                return 'en'
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'en'  # Default to English if detection fails
    
    def translate_to_english(self, text):
        """Translate Swahili text to English"""
        try:
            translation = self.translator.translate(text, src='sw', dest='en')
            return translation.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def translate_to_swahili(self, text):
        """Translate English text to Swahili"""
        try:
            translation = self.translator.translate(text, src='en', dest='sw')
            return translation.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def preprocess_text(self, text, language='en'):
        """Text preprocessing based on language"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback simple tokenization
            tokens = text.split()
        
        # Remove stopwords based on language
        if language == 'sw':
            filtered_tokens = [word for word in tokens if word not in self.swahili_stop_words and len(word) > 2]
        else:
            filtered_tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return filtered_tokens
    
    def get_all_keywords(self):
        """Get all keywords from our dictionaries"""
        all_keywords = set()
        for keywords in self.intent_keywords.values():
            all_keywords.update(keywords)
        for crop_variants in self.crop_entities.values():
            all_keywords.update(crop_variants)
        for pest_variants in self.pest_entities.values():
            all_keywords.update(pest_variants)
        for symptom_variants in self.symptom_entities.values():
            all_keywords.update(symptom_variants)
        return all_keywords
    
    def extract_intent(self, tokens, language='en'):
        """Extract intent from tokens based on language"""
        if language == 'sw':
            intent_keywords = self.swahili_intent_keywords
        else:
            intent_keywords = self.intent_keywords
        
        intent_scores = {intent: 0 for intent in intent_keywords}
        
        for token in tokens:
            for intent, keywords in intent_keywords.items():
                if token in keywords:
                    intent_scores[intent] += 2

        #check for partial matches
        query_text = ' '.join(tokens)
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if ' ' in keyword and keyword in query_text:
                    # Even higher weight for multi-word matches
                    intent_scores[intent] += 3

        # Special handling for common agricultural patterns
        if any(word in tokens for word in ['price', 'cost', 'market', 'sell', 'buy', 'bei', 'soko', 'uuzaji']):
            intent_scores['market'] += 3
    
        if any(word in tokens for word in ['rain', 'weather', 'sun', 'dry', 'mvua', 'jua', 'ukame', 'baridi']):
            intent_scores['weather'] += 3
    
        if any(word in tokens for word in ['pest', 'insect', 'worm', 'bug', 'wadudu', 'mdudu', 'funza']):
            intent_scores['pest'] += 3
    
        if any(word in tokens for word in ['disease', 'sick', 'rot', 'fungus', 'ugonjwa', 'kuoza', 'kuharibu']):
            intent_scores['disease'] += 3
        
        # Get the intent with the highest score
        if max(intent_scores.values()) > 0:
            primary_intent = max(intent_scores, key=intent_scores.get)
            return primary_intent
        
        return "general"
    
    def extract_entities(self, tokens):
        """Enhanced entity extraction with normalization"""
        entities = {
            'crops': [],
            'pests': [],
            'symptoms': [],
            'other': []
        }
        
        for token in tokens:
            # Check for crops
            for crop_name, variants in self.crop_entities.items():
                if token in variants and crop_name not in entities['crops']:
                    entities['crops'].append(crop_name)
                    
            # Check for pests
            for pest_name, variants in self.pest_entities.items():
                if token in variants and pest_name not in entities['pests']:
                    entities['pests'].append(pest_name)
                    
            # Check for symptoms
            for symptom_name, variants in self.symptom_entities.items():
                if token in variants and symptom_name not in entities['symptoms']:
                    entities['symptoms'].append(symptom_name)
                    
            # Add important terms not categorized above to 'other'
            if (token not in entities['crops'] and
                token not in entities['pests'] and
                token not in entities['symptoms'] and
                token in self.get_all_keywords()):
                entities['other'].append(token)
                
        return entities
    
    def search_knowledge_base(self, intent, entities):
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
    
        try:
            # Build query based on intent and entities
            base_query = """
            SELECT *,
                (CASE WHEN category = %s THEN 3
                    WHEN keywords LIKE %s THEN 2
                  ELSE 1 END) as relevance
            FROM knowledge_article
            WHERE category = %s OR keywords LIKE %s
            """
            params = [intent, f"%{intent}%", intent, f"%{intent}%"]
        
            # Add conditions for each entity
            all_entities = []
            for entity_list in entities.values():
                all_entities.extend(entity_list)
        
            if all_entities:
                entity_conditions = []
                for entity in all_entities:
                    entity_conditions.append("keywords LIKE %s")
                    params.append(f"%{entity}%")
                base_query += " OR " + " OR ".join(entity_conditions)
        
            base_query += " ORDER BY relevance DESC, article_id LIMIT 3"
            
            logger.info(f"Executing query: {base_query}")
            logger.info(f"With parameters: {params}")

            cursor.execute(base_query, params)
            results = cursor.fetchall()

            logger.info(f"Search results: {results}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
        finally:
            cursor.close()

    def calculate_confidence(self, intent, entities, knowledge_items, tokens):
        """Calculate confidence score based on multiple factors"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we found relevant knowledge
        if knowledge_items:
            confidence += 0.3
            
            # Check if the top result has high relevance
            if knowledge_items[0].get('relevance', 0) >= 2:
                confidence += 0.2
        
        # Increase confidence if we detected specific entities
        if entities['crops'] or entities['pests'] or entities['symptoms']:
            confidence += 0.1
            
        # Increase confidence if intent is clear (not general)
        if intent != 'general':
            confidence += 0.1
            
        # Decrease confidence for very short queries
        if len(tokens) < 3:
            confidence -= 0.1
            
        # Ensure confidence is between 0 and 1
        return max(0.1, min(0.95, confidence))
    
    def generate_response(self, knowledge_items, intent, entities, language='en'):
        """Generate responses in the appropriate language WITHOUT feedback request"""
        if not knowledge_items:
            # No results found - escalate to expert
            if language == 'sw':
                return "Sikupata taarifa maalum kuhusu swali lako. Swali lako limewasilishwa kwa wataalamu wa kilimo wataojibu hivi karibuni."
            else:
                return "I couldn't find specific information about your query. Your question has been forwarded to our agricultural experts who will respond shortly."
        
        # Start with a context-aware introduction
        crop_mention = ""
        if entities['crops']:
            crop_mention = f" for {', '.join(entities['crops'])}" if language == 'en' else f" kwa {', '.join(entities['crops'])}"

        if language == 'sw':
            if intent == "pest":
                response = f"Kwa maswala ya wadudu{crop_mention}: "
            elif intent == "disease":
                response = f"Kwa matatizo ya magonjwa{crop_mention}: "
            elif intent == "soil":
                response = f"Kwa usimamizi wa udongo{crop_mention}: "
            elif intent == "weather":
                response = f"Kwa ushauri wa hali ya hewa{crop_mention}: "
            elif intent == "crop":
                response = f"Kwa kilimo{crop_mention}: "
            elif intent == "water":
                response = f"Kwa usimamizi wa maji{crop_mention}: "
            elif intent == "market":
                response = f"Kwa taarifa za soko{crop_mention}: "
            else:
                response = f"Haya ni ushauri wa kilimo{crop_mention}: "
        else:
            if intent == "pest":
                response = f"For pest issues{crop_mention}: "
            elif intent == "disease":
                response = f"For disease problems{crop_mention}: "
            elif intent == "soil":
                response = f"For soil management{crop_mention}: "
            elif intent == "weather":
                response = f"For weather-related advice{crop_mention}: "
            elif intent == "crop":
                response = f"For crop cultivation{crop_mention}: "
            elif intent == "water":
                response = f"For irrigation guidance{crop_mention}: "
            elif intent == "market":
                response = f"For market information{crop_mention}: "
            else:
                response = f"Here's some agricultural advice{crop_mention}: "

        # Add the knowledge base content
        for i, item in enumerate(knowledge_items):
            if i > 0:
                response += " " + ("Also: " if language == 'en' else "Pia: ")
            
            # Use the appropriate language content
            if language == 'sw' and 'content_sw' in item and item['content_sw']:
                response += f"{item['content_sw']}"
            else:
                response += f"{item['content']}"
            
            # Limit response length for SMS
            if len(response) > 140 and i >= 1:
                response += " [More info available from experts]" if language == 'en' else " [Maelezo zaidi yanapatikana kwa wataalamu]"
                break
    
        return response
    
    def classify_region(self, location):
        """Classify a location into a Kenyan region"""
        if not location or location == 'Unknown':
            return None
    
        location_lower = location.lower()
    
        # Central Kenya regions
        central_keywords = {'kiambu', 'muranga', 'nyeri', 'kirinyaga', 'nyandarua', 'emubu'}
        if any(keyword in location_lower for keyword in central_keywords):
            return 'central'
    
        # Western Kenya regions
        western_keywords = {'kisumu', 'kakamega', 'bungoma', 'busia', 'vihiga', 'siaya', 'homabay', 'migori', 'kisii', 'nyamira', 'kericho', 'bomet'}
        if any(keyword in location_lower for keyword in western_keywords):
            return 'western'
    
        # Rift Valley regions
        rift_keywords = {'nakuru', 'eldoret', 'uasin gishu', 'nandi', 'trans nzoia', 'baringo', 'laikipia', 'samburu', 'narok', 'kajiado', 'pokot'}
        if any(keyword in location_lower for keyword in rift_keywords):
            return 'rift'
    
        # Eastern Kenya regions
        eastern_keywords = {'machakos', 'makueni', 'kitui', 'meru', 'tharaka', 'embu'}
        if any(keyword in location_lower for keyword in eastern_keywords):
            return 'eastern'
    
        # Coastal regions
        coastal_keywords = {'mombasa', 'kilifi', 'kwale', 'lamu', 'taita', 'taveta'}
        if any(keyword in location_lower for keyword in coastal_keywords):
            return 'coastal'
    
        # ASAL regions
        asal_keywords = {'garissa', 'wajir', 'mandera', 'marsabit', 'isiolo', 'turkana'}
        if any(keyword in location_lower for keyword in asal_keywords):
            return 'asal'
    
        return None
    
    def extract_location_from_query(self, text):
        """Extract potential location hints from the query text"""
        # Common Kenyan counties and regions
        kenyan_locations = {
            'nairobi', 'mombasa', 'kisumu', 'nakuru', 'eldoret', 'thika', 'meru', 'nyeri',
            'machakos', 'kitui', 'embu', 'kakamega', 'kisii', 'bomet', 'kericho', 'bungoma',
            'busia', 'siaya', 'homa bay', 'migori', 'kajiado', 'narok', 'trans nzoia', 'uasin gishu',
            'nyandarua', 'kiambu', 'muranga', 'kirinyaga', 'nyamira', 'vihiga', 'baringo', 'laikipia',
            'nakuru', 'makueni', 'taita taveta', 'tana river', 'lamu', 'kilifi', 'kwale', 'garissa',
            'wajir', 'mandera', 'marsabit', 'isiolo', 'west pokot', 'samburu', 'turbo', 'kitale'
        }
    
        text_lower = text.lower()
        words = set(text_lower.split())
    
        # Check for location mentions
        found_locations = words.intersection(kenyan_locations)
        if found_locations:
            return ', '.join(found_locations)
    
        return None
    
    def process_query(self, text):
        """Main method to process a farmer's query with language detection"""
        # Detect language
        language = self.detect_language(text)
        logger.info(f"Detected language: {language}")
        
        # Translate to English if needed for processing
        if language == 'sw':
            english_text = self.translate_to_english(text)
            logger.info(f"Translated to English: {english_text}")
            processed_text = english_text
        else:
            processed_text = text
        
        # Preprocess text
        tokens = self.preprocess_text(processed_text, language)
        
        # Extract intent and entities
        intent = self.extract_intent(tokens, language)
        entities = self.extract_entities(tokens)
        
        logger.info(f"Processed query: language={language}, intent={intent}, entities={entities}")
        
        # Search knowledge base
        knowledge_items = self.search_knowledge_base(intent, entities)
        
        # Calculate confidence based on multiple factors
        confidence = self.calculate_confidence(intent, entities, knowledge_items, tokens)
        
        # Generate response in the appropriate language
        response = self.generate_response(knowledge_items, intent, entities, language)
        
        return response, intent, entities, language, confidence