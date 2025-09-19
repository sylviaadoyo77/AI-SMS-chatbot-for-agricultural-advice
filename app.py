#app.py
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
import mysql.connector
import mysql.connector.pooling
import os
from dotenv import load_dotenv
import logging
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import uuid
from nlp_processor import NLPProcessor
import datetime
import phonenumbers
from phonenumbers import PhoneNumberFormat
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import traceback
from contextlib import contextmanager
import re
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the Flask application instance
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-for-testing")


# Initialize NLP processor
nlp_processor = NLPProcessor()

# Add this with your other configuration variables
SUPER_ADMIN_REGISTRATION_CODE = os.getenv("SUPER_ADMIN_REGISTRATION_CODE", "agrichatbot@2025")

# Database connection pool
db_pool = None

def init_db_pool():
    global db_pool
    try:
        db_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="agri_pool",
            pool_size=5,
            pool_reset_session=True,
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            autocommit=True
        )
        logger.info("Database connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        db_pool = None

# Initialize the connection pool
init_db_pool()

# Replace your get_db_connection function with this:
def get_db_connection():
    try:
        return mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            autocommit=True
        )
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

@contextmanager
def get_db_cursor():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("Failed to get database connection")
        cursor = conn.cursor(dictionary=True)
        yield cursor
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise e
    finally:
        if cursor:
            try:
                cursor.fetchall()
            except:
                pass # Ignore errors if there no results to fetch
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def admin_required(required_role='admin'):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'admin_id' not in session:
                return redirect(url_for('admin_login'))
            
            # Check if user has the required role
            user_role = session.get('admin_role', '')
            if required_role == 'super_admin' and user_role != 'super_admin':
                return "Access denied: Super admin required", 403
            elif required_role == 'admin' and user_role not in ['admin', 'super_admin']:
                return "Access denied: Admin required", 403
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Initialize Twilio client
try:
    twilio_client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))
    logger.info("Twilio client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Twilio client: {e}")
    twilio_client = None

# After initializing twilio_client, add:
if twilio_client:
    try:
        # Test the credentials
        account = twilio_client.api.accounts(os.getenv('TWILIO_ACCOUNT_SID')).fetch()
        logger.info(f"Twilio account verified: {account.friendly_name}")
    except Exception as e:
        logger.error(f"Twilio authentication failed: {e}")
        twilio_client = None
# decorator to protect routes based on admin roles
def admin_required(required_role='admin'):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'admin_id' not in session:
                return redirect(url_for('admin_login'))
            
            # Check if user has the required role
            user_role = session.get('admin_role', '')
            if required_role == 'super_admin' and user_role != 'super_admin':
                return "Access denied: Super admin required", 403
            elif required_role == 'admin' and user_role not in ['admin', 'super_admin']:
                return "Access denied: Admin required", 403
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Phone number formatting function
def format_phone_number(phone_number):
    """Format phone number to E.164 format"""
    try:
        parsed = phonenumbers.parse(phone_number, None)
        return phonenumbers.format_number(parsed, PhoneNumberFormat.E164)
    except:
        # If parsing fails, try to clean up the number
        cleaned = ''.join(filter(str.isdigit, phone_number))
        if cleaned.startswith('0'):
            cleaned = '+254' + cleaned[1:]  # Assuming Kenyan numbers
        return cleaned

def save_to_knowledge_base(response_text, query_id, keywords="", category="expert_advice"):
    """Save expert responses to the knowledge base for future AI learning"""
    try:
        with get_db_cursor() as cursor:
            # Generate a unique article ID
            article_id = f"ART_{uuid.uuid4().hex[:10]}"
            
            # If no keywords provided, extract them from the response
            if not keywords:
                # Simple keyword extraction from response text
                important_words = ['maize', 'wheat', 'rice', 'beans', 'pest', 'disease', 
                                  'soil', 'water', 'fertilizer', 'weather', 'harvest']
                found_keywords = []
                for word in important_words:
                    if word in response_text.lower():
                        found_keywords.append(word)
                keywords = ','.join(found_keywords) if found_keywords else 'general_advice'
            
            # Insert into knowledge base
            cursor.execute(
                "INSERT INTO knowledge_article (article_id, title, content, keywords, category) VALUES (%s, %s, %s, %s, %s)",
                (article_id, f"Expert Response to Query {query_id}", response_text, keywords, category)
            )
            
            logger.info(f"Saved expert response to knowledge base with ID: {article_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error saving to knowledge base: {e}")
        return False
    
# Add this method to your NLPProcessor class
def extract_keywords_from_text(self, text, num_keywords=5):
    """Extract potential keywords from text for knowledge base tagging"""
    try:
        # Simple keyword extraction based on frequency
        words = text.lower().split()
        word_count = {}
        
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)
            if len(word) > 3 and word not in self.stop_words and word not in self.swahili_stop_words:
                word_count[word] = word_count.get(word, 0) + 1
        
        # Get most frequent words
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:num_keywords]]
        
        return ','.join(keywords)
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return "expert_advice"
    
@app.route('/expert/knowledge', methods=['GET', 'POST'])
def manage_knowledge():
    """Standalone interface for experts to manage the knowledge base"""
    if 'expert_id' not in session:
        return redirect(url_for('expert_login'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        keywords = request.form.get('keywords')
        category = request.form.get('category', 'expert_advice')
        
        if not all([title, content]):
            flash('Title and content are required.', 'error')
            return redirect(url_for('manage_knowledge'))
        
        try:
            with get_db_cursor() as cursor:
                article_id = f"ART_{uuid.uuid4().hex[:10]}"
                cursor.execute(
                    "INSERT INTO knowledge_article (article_id, title, content, keywords, category) VALUES (%s, %s, %s, %s, %s)",
                    (article_id, title, content, keywords, category)
                )
                flash('Knowledge article added successfully!', 'success')
        except Exception as e:
            logger.error(f"Error adding knowledge article: {e}")
            flash('Error adding knowledge article.', 'error')
    
    # Display existing knowledge articles
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM knowledge_article ORDER BY article_id DESC LIMIT 20")
            articles = cursor.fetchall()
    except Exception as e:
        logger.error(f"Error fetching knowledge articles: {e}")
        articles = []
    
    return render_template('manage_knowledge.html', articles=articles)


# Make session permanent
@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=30)

# Basic health check route
@app.route('/')
def home():
    return render_template('index.html')

# Health check route
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

# Database health check
@app.route('/health/database')
def database_health():
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            return jsonify({"status": "healthy", "database": "connected"})
    except Exception as e:
        return jsonify({"status": "unhealthy", "database": "disconnected", "error": str(e)}), 500
    

# SMS webhook endpoint for Twilio
@app.route('/sms', methods=['POST'])
def sms_reply():
    """Endpoint for Twilio to forward incoming SMS messages to."""
    # Get the message from the request
    incoming_msg = request.form.get('Body', '').strip()
    from_number = request.form.get('From', '')
    formatted_number = format_phone_number(from_number)
    logger.info(f"Received SMS from {formatted_number}: {incoming_msg}")

    # Start our TwiML response
    resp = MessagingResponse()

    try:
        with get_db_cursor() as cursor:
            # 1. Check if farmer is registered and get their state
            cursor.execute("SELECT farmer_id, conversation_state, location FROM farmer WHERE phone_number = %s", (formatted_number,))
            farmer = cursor.fetchone()

            farmer_id = None
            conversation_state = 'normal'
            current_location = 'Unknown'
        
            if farmer:
                farmer_id = farmer['farmer_id']
                conversation_state = farmer['conversation_state'] or 'normal'
                current_location = farmer['location'] or 'Unknown'
                logger.info(f"Found existing farmer: {farmer_id} with state: {conversation_state}, location: {current_location}")
            else:
                # Register new farmer
                farmer_id = f"FARMER_{uuid.uuid4().hex[:10]}"
            
                # Try to extract location from the query if it contains location hints
                detected_location = nlp_processor.extract_location_from_query(incoming_msg)
                if detected_location:
                    current_location = detected_location
                else:
                    current_location = 'Unknown'
            
                cursor.execute(
                    "INSERT INTO farmer (farmer_id, phone_number, location, conversation_state) VALUES (%s, %s, %s, %s)",
                    (farmer_id, formatted_number, current_location, 'normal')
                )
                logger.info(f"Registered new farmer: {farmer_id} with location: {current_location}")
            
                # If location is unknown, ask for it
                if current_location == 'Unknown':
                    cursor.execute(
                        "UPDATE farmer SET conversation_state = 'awaiting_location' WHERE farmer_id = %s",
                        (farmer_id,)
                    )
                    
                    # Detect language for appropriate response
                    language = nlp_processor.detect_language(incoming_msg)
                    location_request = "Karibu kwa huduma yetu ya kilimo! Tafadhali tueleke eneo lako (kaunti au wilaya) ili tukupe ushauri unaofaa."
                    if language == 'en':
                        location_request = "Welcome to our agricultural service! Please tell us your location (county or district) so we can provide appropriate advice."
                    
                    resp.message(location_request)
                    return str(resp)

            # 2. Handle different conversation states
            if conversation_state == 'awaiting_feedback':
                # Process feedback
                feedback_id = f"FB_{uuid.uuid4().hex[:10]}"
                
                # Get the most recent query for feedback association
                cursor.execute("""
                    SELECT query_id FROM query 
                    WHERE farmer_id = %s 
                    ORDER BY query_timestamp DESC 
                    LIMIT 1
                """, (farmer_id,))
                
                recent_query = cursor.fetchone()
                
                if recent_query:
                    query_id = recent_query['query_id']
                    
                    # Determine rating based on response
                    if incoming_msg.lower() in ['yes', 'ndio', 'sawa', 'thank you', 'asante']:
                        rating = 5
                        feedback_msg = "Asante kwa maoni yako! Tutajaribu kuboresha huduma yetu zaidi." if any(word in incoming_msg.lower() for word in ['ndio', 'asante']) else "Thank you for your feedback! We'll continue to improve our service."
                    elif incoming_msg.lower() in ['no', 'hapana', 'si sawa']:
                        rating = 1
                        feedback_msg = "Pole kwa kutokukuhudumia vyema. Tutaongeza juhudi zetu." if any(word in incoming_msg.lower() for word in ['hapana', 'si sawa']) else "Sorry we couldn't help better. We'll work to improve."
                    else:
                        rating = 3
                        feedback_msg = "Asante kwa maoni yako." if any(word in incoming_msg.lower() for word in ['asante', 'shukran']) else "Thank you for your feedback."
                    
                    # Store feedback
                    cursor.execute(
                        "INSERT INTO feedback (feedback_id, query_id, farmer_id, rating) VALUES (%s, %s, %s, %s)",
                        (feedback_id, query_id, farmer_id, rating)
                    )
                    
                    # Reset conversation state
                    cursor.execute(
                        "UPDATE farmer SET conversation_state = 'normal' WHERE farmer_id = %s",
                        (farmer_id,)
                    )
                    
                    resp.message(feedback_msg)
                else:
                    resp.message("Sorry, we couldn't find a recent query to associate with your feedback.")
                    # Reset conversation state
                    cursor.execute(
                        "UPDATE farmer SET conversation_state = 'normal' WHERE farmer_id = %s",
                        (farmer_id,)
                    )
                
                return str(resp)
            
            elif conversation_state == 'awaiting_location':
                # Process location response
                detected_location = nlp_processor.extract_location_from_query(incoming_msg)
                if detected_location:
                    new_location = detected_location
                else:
                    # Use the input as location if no specific location detected
                    new_location = incoming_msg
                
                cursor.execute(
                    "UPDATE farmer SET location = %s, conversation_state = 'normal' WHERE farmer_id = %s",
                    (new_location, farmer_id)
                )
                logger.info(f"Updated farmer {farmer_id} location to: {new_location}")
                
                # Confirm location update
                language = nlp_processor.detect_language(incoming_msg)
                confirmation_msg = f"Asante! Eneo lako limehifadhiwa kama {new_location}. Sasa unaweza kuuliza swali lolote kuhusu kilimo."
                if language == 'en':
                    confirmation_msg = f"Thank you! Your location has been saved as {new_location}. You can now ask any agricultural question."
                
                resp.message(confirmation_msg)
                return str(resp)
            
            # 3. Normal query processing (for conversation_state = 'normal')
            # Log the query in the database with status 'pending'
            query_id = f"QUERY_{uuid.uuid4().hex[:10]}"
            cursor.execute(
                "INSERT INTO query (query_id, farmer_id, query_text, status) VALUES (%s, %s, %s, %s)",
                (query_id, farmer_id, incoming_msg, 'pending')
            )

            # Process the query with enhanced NLP
            ai_response, intent, entities, language, confidence = nlp_processor.process_query(incoming_msg)

            # Update query status based on confidence score
            if confidence < 0.6:
                new_status = 'escalated'
                logger.info(f"Query {query_id} escalated to experts due to low confidence: {confidence}")

                if language == 'sw':
                    ai_response = "Shida yako imewasilishwa kwa wataalamu wa kilimo. Utapokea jibu hivi karibuni."
                else:
                    ai_response = "Your query has been forwarded to our agricultural experts. You will receive a response shortly."
            else:
                new_status = 'answered_ai'
                logger.info(f"Query {query_id} answered by AI with confidence: {confidence}")

            # Update query status based on response content
            if "expert" in ai_response.lower() and new_status != 'escalated':
                new_status = 'escalated'
                logger.info(f"Query {query_id} escalated to experts")
            
            cursor.execute(
                "UPDATE query SET status = %s WHERE query_id = %s",
                (new_status, query_id)
            )
            
            # 4. Add feedback request for successful responses
            if new_status == 'answered_ai':
                # Get the farmer's location for personalized response
                cursor.execute("SELECT location FROM farmer WHERE farmer_id = %s", (farmer_id,))
                farmer_data = cursor.fetchone()
                farmer_location = farmer_data['location'] if farmer_data and farmer_data['location'] != 'Unknown' else 'Unknown'
                
                if language == 'sw':
                    ai_response += " Je, jibu hili likusaidia? Tuma 'Ndio' ou 'Hapana'."
                else:
                    ai_response += " Did this answer help? Reply 'Yes' or 'No'."
                
                # Set conversation state to awaiting feedback
                cursor.execute(
                    "UPDATE farmer SET conversation_state = 'awaiting_feedback' WHERE farmer_id = %s",
                    (farmer_id,)
                )

            # 5. Send the AI-generated response
            resp.message(ai_response)

    except Exception as e:
        logger.error(f"Error processing SMS: {e}")
        logger.error(traceback.format_exc())
        # Send a generic error message to the user
        resp.message("Sorry, we're experiencing technical difficulties. Please try again later.")

    return str(resp)

# API endpoint to test NLP processing without SMS
@app.route('/api/process-query', methods=['POST'])
def api_process_query():
    """API endpoint to test NLP processing"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    query_text = data['query']
    try:
        response, intent, entities, language, confidence = nlp_processor.process_query(query_text)
        return jsonify({
            'query': query_text, 
            'response': response,
            'intent': intent,
            'entities': entities,
            'language': language,
            'confidence': confidence
        })
    except Exception as e:
        logger.error(f"Error in API processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# API endpoint to get query statistics
@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get statistics about queries and responses"""
    try:
        with get_db_cursor() as cursor:
            # Get total queries
            cursor.execute("SELECT COUNT(*) as total FROM query")
            total_queries = cursor.fetchone()['total']
            
            # Get queries by status
            cursor.execute("SELECT status, COUNT(*) as count FROM query GROUP BY status")
            status_counts = cursor.fetchall()
            
            # Get recent queries
            cursor.execute("SELECT query_text, status, query_timestamp FROM query ORDER BY query_timestamp DESC LIMIT 10")
            recent_queries = cursor.fetchall()
            
            return jsonify({
                'total_queries': total_queries,
                'status_counts': status_counts,
                'recent_queries': recent_queries
            })
            
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Dashboard route
@app.route('/dashboard')
def dashboard():
    """Simple dashboard to view query statistics"""
    try:
        with get_db_cursor() as cursor:
            # Get total queries
            cursor.execute("SELECT COUNT(*) as total FROM query")
            total_queries = cursor.fetchone()['total']
            
            # Get queries by status
            cursor.execute("SELECT status, COUNT(*) as count FROM query GROUP BY status")
            status_counts = cursor.fetchall()
            
            # Get recent queries
            cursor.execute("SELECT query_text, status, query_timestamp FROM query ORDER BY query_timestamp DESC LIMIT 10")
            recent_queries = cursor.fetchall()
            
            statistics = {
                'total_queries': total_queries,
                'status_counts': status_counts,
                'recent_queries': recent_queries
            }
            
            return render_template('dashboard.html', statistics=statistics)
            
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return f"Error loading dashboard: {e}"

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        user_type = request.form.get('user_type')
        username = request.form.get('username')
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        registration_code = request.form.get('registration_code', '')
        
        # Add detailed logging
        logger.info(f"Signup attempt - Type: {user_type}, Username: {username}, Email: {email}")
        
        # Basic validation
        if not all([user_type, username, full_name, email, password, confirm_password]):
            logger.warning("Signup failed: Missing required fields")
            return render_template('signup.html', error="All fields are required")
        
        if password != confirm_password:
            logger.warning("Signup failed: Passwords don't match")
            return render_template('signup.html', error="Passwords do not match")
        
        if len(password) < 8:
            logger.warning("Signup failed: Password too short")
            return render_template('signup.html', error="Password must be at least 8 characters")
        
        # Super admin registration code validation
        if user_type == 'super_admin':
            if registration_code != os.getenv('SUPER_ADMIN_REGISTRATION_CODE', 'agrichatbot2025'):
                logger.warning("Super admin registration failed: Invalid registration code")
                return render_template('signup.html', error="Invalid super admin registration code")
        
        try:
            # Hash the password
            hashed_password = generate_password_hash(password)
            logger.info(f"Password hashed successfully: {hashed_password[:50]}...")
            
            with get_db_cursor() as cursor:
                if user_type == 'expert':
                    # Check if expert already exists
                    cursor.execute("SELECT expert_id FROM agricultural_expert WHERE expert_name = %s", (username,))
                    existing_expert = cursor.fetchone()
                    if existing_expert:
                        logger.warning(f"Signup failed: Expert username already exists - {username}")
                        return render_template('signup.html', error="Expert username already exists")
                    
                    # Generate expert ID
                    expert_id = f"EXP_{uuid.uuid4().hex[:8]}"
                    logger.info(f"Generated expert ID: {expert_id}")

                    cursor.execute("SHOW COLUMNS FROM agricultural_expert LIKE 'is_active'")
                    has_is_active = cursor.fetchone() is not None
                    if has_is_active:
                        # Insert new expert with is_active column
                        cursor.execute(
                            "INSERT INTO agricultural_expert (expert_id, expert_name, contact_info, password_hash, is_active) VALUES (%s, %s, %s, %s, %s)",
                            (expert_id, username, email, hashed_password, 1)
                        )
                    else:
                        # Insert without is_active column
                        cursor.execute(
                            "INSERT INTO agricultural_expert (expert_id, expert_name, contact_info, password_hash) VALUES (%s, %s, %s, %s)",
                            (expert_id, username, email, hashed_password)
                        )

                    logger.info("Expert inserted into database successfully")
                    return render_template('signup_success.html', 
                           message='Expert account created successfully. You can now log in.')
                
                elif user_type == 'admin':
                    # Check if admin already exists
                    cursor.execute("SELECT id FROM admin_users WHERE username = %s", (username,))
                    existing_admin = cursor.fetchone()
                    if existing_admin:
                        logger.warning(f"Signup failed: Admin username already exists - {username}")
                        return render_template('signup.html', error="Admin username already exists")
                    
                    # Generate admin ID
                    admin_id = f"ADM_{uuid.uuid4().hex[:8]}"
                    logger.info(f"Generated admin ID: {admin_id}")
                    
                    # Insert new admin (requires approval)
                    cursor.execute(
                        "INSERT INTO admin_users (id, username, full_name, email, password_hash, role, is_active, registration_status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (admin_id, username, full_name, email, hashed_password, 'admin', 0, 'pending')
                    )
                    logger.info("Admin inserted into database successfully")
                    
                    return render_template('signup_success.html', 
                           message='Admin account created successfully. It will be activated after approval by a super admin.')
                
                elif user_type == 'super_admin':
                    # Check if username already exists (not role-based)
                    cursor.execute("SELECT id FROM admin_users WHERE username = %s", (username,))
                    existing_user = cursor.fetchone()
                    if existing_user:
                        logger.warning(f"Signup failed: Username already exists - {username}")
                        return render_template('signup.html', error="Username already exists")
                    
                    # Generate super admin ID
                    admin_id = f"ADM_{uuid.uuid4().hex[:8]}"
                    logger.info(f"Generated super admin ID: {admin_id}")
                    
                    # Insert new super admin (auto-approved)
                    cursor.execute(
                        "INSERT INTO admin_users (id, username, full_name, email, password_hash, role, is_active, registration_status, approved_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (admin_id, username, full_name, email, hashed_password, 'super_admin', 1, 'approved', 'system')
                    )
                    logger.info("Super admin inserted into database successfully")
                    
                    return render_template('signup_success.html', 
                           message='Super Admin account created successfully. You can now log in.')
                
                else:
                    logger.warning(f"Signup failed: Invalid user type - {user_type}")
                    return render_template('signup.html', error="Invalid user type")
                    
        except Exception as e:
            logger.error(f"Error during signup: {str(e)}")
            logger.error(traceback.format_exc())  
            return render_template('signup.html', error="Error creating account. Please try again.")
    
    return render_template('signup.html')

@app.route('/signup/success')
def signup_success():
    message = request.args.get('message', 'Account created successfully.')
    return render_template('signup_success.html', message=message)

# Route to approve a user registration
@app.route('/admin/approve-user/<user_type>/<user_id>')
@admin_required('super_admin')
def approve_user(user_type, user_id):
    try:
        with get_db_cursor() as cursor:
            if user_type == 'admin':
                # Check if registration_status column exists
                cursor.execute("SHOW COLUMNS FROM admin_users LIKE 'registration_status'")
                has_registration_status = cursor.fetchone() is not None
                
                if has_registration_status:
                    cursor.execute(
                        "UPDATE admin_users SET is_active = 1, registration_status = 'approved', approved_by = %s, approval_date = CURRENT_TIMESTAMP WHERE id = %s",
                        (session.get('admin_id'), user_id)
                    )
                else:
                    cursor.execute(
                        "UPDATE admin_users SET is_active = 1 WHERE id = %s",
                        (user_id,)
                    )
            elif user_type == 'expert':
                # Check if is_active column exists
                cursor.execute("SHOW COLUMNS FROM agricultural_expert LIKE 'is_active'")
                has_is_active = cursor.fetchone() is not None
                
                if has_is_active:
                    cursor.execute(
                        "UPDATE agricultural_expert SET is_active = 1 WHERE expert_id = %s",
                        (user_id,)
                    )
                else:
                    # If is_active column doesn't exist, we can't mark as active
                    return "Cannot approve expert - missing is_active column", 500
            else:
                return "Invalid user type", 400
            
            return redirect(url_for('admin_approvals'))
            
    except Exception as e:
        logger.error(f"Error approving user: {e}")
        return "Error approving user", 500

# Route to reject a user registration
@app.route('/admin/reject-user/<user_type>/<user_id>')
@admin_required('super_admin')
def reject_user(user_type, user_id):
    try:
        with get_db_cursor() as cursor:
            if user_type == 'admin':
                cursor.execute(
                    "UPDATE admin_users SET registration_status = 'rejected', approved_by = %s, approval_date = CURRENT_TIMESTAMP WHERE id = %s",
                    (session.get('admin_id'), user_id)
                )
            elif user_type == 'expert':
                cursor.execute(
                    "DELETE FROM agricultural_expert WHERE expert_id = %s",
                    (user_id,)
                )
            else:
                return "Invalid user type", 400
            
            return redirect(url_for('admin_approvals'))
            
    except Exception as e:
        logger.error(f"Error rejecting user: {e}")
        return "Error rejecting user", 500

# Expert login route
@app.route('/expert/login', methods=['GET', 'POST'])
def expert_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        try:
            with get_db_cursor() as cursor:
                # Get expert from database
                cursor.execute(
                    "SELECT expert_id, expert_name, password_hash FROM agricultural_expert WHERE expert_name = %s", 
                    (username,)
                )
                expert = cursor.fetchone()
                
                # Verify password
                if expert and check_password_hash(expert['password_hash'], password):
                    session['expert_username'] = expert['expert_name']
                    session['expert_id'] = expert['expert_id']
                    return redirect(url_for('expert_dashboard'))
                else:
                    return render_template('expert_login.html', error="Invalid credentials")
                    
        except Exception as e:
            logger.error(f"Error during expert login: {e}")
            return render_template('expert_login.html', error="Login error. Please try again.")
    
    return render_template('expert_login.html')

# Expert dashboard route
@app.route('/expert/dashboard')
def expert_dashboard():
    if 'expert_username' not in session:
        return redirect(url_for('expert_login'))
    
    try:
        with get_db_cursor() as cursor:
            # Get escalated queries with farmer location
            cursor.execute("""
                SELECT q.query_id, q.query_text, q.query_timestamp, f.phone_number, f.location 
                FROM query q 
                JOIN farmer f ON q.farmer_id = f.farmer_id 
                WHERE q.status = 'escalated' 
                ORDER BY q.query_timestamp DESC
            """)
            escalated_queries = cursor.fetchall()
            
            # Get recently answered queries
            cursor.execute("""
                SELECT q.query_id, q.query_text, r.response_text, r.response_timestamp, e.expert_name 
                FROM query q 
                JOIN response r ON q.query_id = r.query_id 
                JOIN agricultural_expert e ON r.expert_id = e.expert_id 
                WHERE r.response_type = 'expert' 
                ORDER BY r.response_timestamp DESC 
                LIMIT 10
            """)
            answered_queries = cursor.fetchall()
            
            return render_template('expert_dashboard.html', 
                                 escalated_queries=escalated_queries,
                                 answered_queries=answered_queries,
                                 expert_username=session['expert_username'])
            
    except Exception as e:
        logger.error(f"Error loading expert dashboard: {e}")
        return f"Error loading dashboard: {e}"

# Expert response route
@app.route('/expert/respond/<query_id>', methods=['POST'])
def expert_respond(query_id):
    if 'expert_id' not in session:
        return redirect(url_for('expert_login'))
    
    response_text = request.form.get('response')
    keywords = request.form.get('keywords', '')
    category = request.form.get('category', 'general')
    
    if not response_text:
        return "Response text is required", 400
    
    try:
        with get_db_cursor() as cursor:
            # Get expert ID
            expert_id = session.get('expert_id')
            
            if not expert_id:
                return "Expert not authenticated properly", 401
            
            # Create response
            response_id = f"RESP_{uuid.uuid4().hex[:10]}"
            logger.info(f"Creating response {response_id} for query {query_id}")
            
            # Insert response
            cursor.execute(
                "INSERT INTO response (response_id, query_id, response_text, response_type, expert_id) VALUES (%s, %s, %s, %s, %s)",
                (response_id, query_id, response_text, 'expert', expert_id)
            )
            
            # Update query status - Fixed parameter passing
            cursor.execute(
                "UPDATE query SET status = %s, response_id = %s WHERE query_id = %s",
                ['answered_expert', response_id, query_id]  # Use list instead of set
            )
            
            # Add to knowledge base if keywords provided
            if keywords:
                article_id = f"ART_{uuid.uuid4().hex[:10]}"
                cursor.execute(
                    "INSERT INTO knowledge_article (article_id, title, content, keywords, category) VALUES (%s, %s, %s, %s, %s)",
                    [article_id, f"Expert Advice for {query_id}", response_text, keywords, category]  # Use list
                )
            
            # Send SMS response to farmer
            cursor.execute("""
                SELECT f.phone_number, q.query_text 
                FROM farmer f 
                JOIN query q ON f.farmer_id = q.farmer_id 
                WHERE q.query_id = %s
            """, [query_id])  # Use list
            
            query_info = cursor.fetchone()
            
            if query_info and twilio_client:
                try:
                    # Truncate response if too long for SMS
                    if len(response_text) > 140:
                        response_text = response_text[:137] + "..."
                    
                    message = twilio_client.messages.create(
                        body=f"Expert response: {response_text}",
                        from_=os.getenv('TWILIO_PHONE_NUMBER'),
                        to=query_info['phone_number']
                    )
                    logger.info(f"Expert response sent to {query_info['phone_number']}, Message SID: {message.sid}")
                except Exception as e:
                    logger.error(f"Failed to send expert response via SMS: {e}")
            
            return redirect(url_for('expert_dashboard'))
            
    except Exception as e:
        logger.error(f"Error processing expert response: {e}")
        logger.error(traceback.format_exc())
        return f"Error processing response: {e}", 500
# Expert logout route
@app.route('/expert/logout')
def expert_logout():
    session.pop('expert_id', None)
    session.pop('expert_name', None)
    return redirect(url_for('expert_login'))

# Super Admin login route
@app.route('/superadmin/login', methods=['GET', 'POST'])
def super_admin_login():
    if 'admin_id' in session and session.get('admin_role') == 'super_admin':
        return redirect(url_for('super_admin_dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        try:
            with get_db_cursor() as cursor:
                # Get admin from database
                cursor.execute(
                    "SELECT id, username, password_hash, full_name, role, is_active, registration_status FROM admin_users WHERE username = %s AND role = 'super_admin'", 
                    (username,)
                )
                admin = cursor.fetchone()
                
                if admin:
                    # For super admin, bypass registration status check
                    if not admin['is_active']:
                        return render_template('super_admin_login.html', error="Super admin account is deactivated")
                    
                    # Verify password
                    if check_password_hash(admin['password_hash'], password):
                        # Update last login timestamp
                        cursor.execute(
                            "UPDATE admin_users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                            (admin['id'],)
                        )
                        
                        # Set session variables
                        session['admin_id'] = admin['id']
                        session['admin_username'] = admin['username']
                        session['admin_full_name'] = admin['full_name']
                        session['admin_role'] = admin['role']
                        
                        return redirect(url_for('super_admin_dashboard'))
                    else:
                        return render_template('super_admin_login.html', error="Invalid credentials")
                else:
                    return render_template('super_admin_login.html', error="Invalid super admin credentials")
                    
        except Exception as e:
            logger.error(f"Error during super admin login: {e}")
            return render_template('super_admin_login.html', error="Login error. Please try again.")
    
    return render_template('super_admin_login.html')

# Update the serve_super_admin route to redirect to the correct login
@app.route('/super_admin.html')
def serve_super_admin_redirect():
    """Redirect from old super_admin.html to new login"""
    return redirect(url_for('super_admin_login'))

@app.route('/admin/verify-super-admin')
def verify_super_admin():
    """Verify the status of the super admin account"""
    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                "SELECT id, username, role, is_active, registration_status FROM admin_users WHERE username = 'superadmin'"
            )
            super_admin = cursor.fetchone()
            
            if super_admin:
                return jsonify({
                    'exists': True,
                    'username': super_admin['username'],
                    'role': super_admin['role'],
                    'is_active': bool(super_admin['is_active']),
                    'registration_status': super_admin['registration_status']
                })
            else:
                return jsonify({'exists': False})
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/admin/debug-super-admin')
def debug_super_admin():
    """Debug route to check the status of super admin accounts"""
    try:
        with get_db_cursor() as cursor:
            # Check for any super admin accounts
            cursor.execute("SELECT * FROM admin_users WHERE role = 'super_admin'")
            super_admins = cursor.fetchall()
            
            return jsonify({
                'super_admin_count': len(super_admins),
                'super_admins': super_admins
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/fix-super-admin', methods=['POST'])
def fix_super_admin():
    """Route to fix super admin account issues"""
    try:
        action = request.form.get('action')
        
        with get_db_cursor() as cursor:
            if action == 'reset_password':
                # Reset password for existing super admin
                new_password = request.form.get('new_password')
                hashed_password = generate_password_hash(new_password)
                
                cursor.execute(
                    "UPDATE admin_users SET password_hash = %s WHERE role = 'super_admin'",
                    (hashed_password,)
                )
                return jsonify({'message': 'Super admin password reset successfully'})
                
            elif action == 'create_new':
                # Create a new super admin account
                username = request.form.get('username')
                email = request.form.get('email')
                password = request.form.get('password')
                hashed_password = generate_password_hash(password)
                
                # Check if username already exists
                cursor.execute("SELECT id FROM admin_users WHERE username = %s", (username,))
                if cursor.fetchone():
                    return jsonify({'error': 'Username already exists'}), 400
                
                # Create new super admin
                admin_id = f"ADM_{uuid.uuid4().hex[:8]}"
                cursor.execute(
                    "INSERT INTO admin_users (id, username, full_name, email, password_hash, role, is_active, registration_status, approved_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (admin_id, username, 'Super Administrator', email, hashed_password, 'super_admin', 1, 'approved', 'system')
                )
                return jsonify({'message': 'New super admin created successfully'})
                
            elif action == 'delete_all':
                # Delete all super admin accounts (use with caution!)
                cursor.execute("DELETE FROM admin_users WHERE role = 'super_admin'")
                return jsonify({'message': 'All super admin accounts deleted'})
                
            else:
                return jsonify({'error': 'Invalid action'}), 400
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@app.route('/admin/super-dashboard')
@admin_required('super_admin')
def super_admin_dashboard():
    """Super Admin Dashboard with enhanced system oversight"""
    try:
        with get_db_cursor() as cursor:
            # Get total queries
            cursor.execute("SELECT COUNT(*) as total FROM query")
            total_queries = cursor.fetchone()['total']
            
            # Get queries by status
            cursor.execute("SELECT status, COUNT(*) as count FROM query GROUP BY status")
            status_counts = cursor.fetchall()
            
            # Convert to a simple dictionary
            status_dict = {}
            for item in status_counts:
                status_dict[item['status']] = item['count']
            
            # Get recent queries
            cursor.execute("""
                SELECT query_id, query_text, status, query_timestamp 
                FROM query 
                ORDER BY query_timestamp DESC 
                LIMIT 10
            """)
            recent_queries = cursor.fetchall()

            # Get super admin statistics
            # Pending admin approvals count
            cursor.execute("SELECT COUNT(*) as count FROM admin_users WHERE registration_status = 'pending' AND role = 'admin'")
            pending_count = cursor.fetchone()['count']
            
            # Total admin count
            cursor.execute("SELECT COUNT(*) as count FROM admin_users WHERE role = 'admin'")
            total_admins = cursor.fetchone()['count']
            
            # Active experts count
            cursor.execute("SELECT COUNT(*) as count FROM agricultural_expert WHERE is_active = 1")
            active_experts = cursor.fetchone()['count']
            
            # Total farmers count
            cursor.execute("SELECT COUNT(*) as count FROM farmer")
            total_farmers = cursor.fetchone()['count']
            
            # Get system usage stats (example - you might need to adjust based on your schema)
            cursor.execute("SELECT COUNT(DISTINCT farmer_id) as active_farmers FROM query WHERE query_timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)")
            active_farmers = cursor.fetchone()['active_farmers']
            
            
            # Prepare statistics
            statistics = {
                'total_queries': total_queries,
                'ai_answered': status_dict.get('answered_ai', 0),
                'escalated': status_dict.get('escalated', 0),
                'expert_answered': status_dict.get('answered_expert', 0),
                'recent_queries': recent_queries,
                'pending_count': pending_count,
                'total_admins': total_admins,
                'active_experts': active_experts,
                'total_farmers': total_farmers,
                'active_farmers': active_farmers
            }
            
            # Add current time for the footer
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return render_template('super_admin_dashboard.html', statistics=statistics, current_time=current_time)
            
    except Exception as e:
        logger.error(f"Error loading super admin dashboard: {e}")
        logger.error(traceback.format_exc())
        return f"Error loading super admin dashboard: {e}"
    
@app.route('/admin/reset-super-admin', methods=['GET', 'POST'])
def reset_super_admin_password():
    """Route to reset the super admin password (for development use only)"""
    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if new_password != confirm_password:
            return render_template('reset_super_admin.html', error="Passwords do not match")
        
        if len(new_password) < 8:
            return render_template('reset_super_admin.html', error="Password must be at least 8 characters")
        
        try:
            hashed_password = generate_password_hash(new_password)
            
            with get_db_cursor() as cursor:
                cursor.execute(
                    "UPDATE admin_users SET password_hash = %s WHERE username = 'superadmin'",
                    (hashed_password,)
                )
                
                return render_template('reset_success.html', 
                                      message='Super admin password reset successfully. You can now log in.')
                
        except Exception as e:
            logger.error(f"Error resetting super admin password: {e}")
            return render_template('reset_super_admin.html', error="Error resetting password")
    
    return render_template('reset_super_admin.html')
    
# Route to serve the super_admin.html page
@app.route('/super_admin.html')
@admin_required('super_admin')
def serve_super_admin():
    """Serve the super admin HTML page"""
    try:
        # Get the statistics data needed for the dashboard
        with get_db_cursor() as cursor:
            # Get total queries
            cursor.execute("SELECT COUNT(*) as total FROM query")
            total_queries = cursor.fetchone()['total']
            
            # Get pending admin approvals count
            cursor.execute("SELECT COUNT(*) as count FROM admin_users WHERE registration_status = 'pending' AND role = 'admin'")
            pending_count = cursor.fetchone()['count']
            
            # Get total admin count
            cursor.execute("SELECT COUNT(*) as count FROM admin_users WHERE role = 'admin'")
            total_admins = cursor.fetchone()['count']
            
            # Get active expert counts
            cursor.execute("SELECT COUNT(*) as count FROM agricultural_expert WHERE is_active = 1")
            active_experts = cursor.fetchone()['count']
            
            # Get queries by status
            cursor.execute("SELECT status, COUNT(*) as count FROM query GROUP BY status")
            status_counts = cursor.fetchall()
            
            # Convert to a simple dictionary
            status_dict = {}
            for item in status_counts:
                status_dict[item['status']] = item['count']
            
            statistics = {
                'total_queries': total_queries,
                'pending_count': pending_count,
                'total_admins': total_admins,
                'active_experts': active_experts,
                'ai_answered': status_dict.get('answered_ai', 0),
                'escalated': status_dict.get('escalated', 0),
                'expert_answered': status_dict.get('answered_expert', 0)
            }
            
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return render_template('super_admin.html', statistics=statistics, current_time=current_time)
            
    except Exception as e:
        logger.error(f"Error serving super_admin.html: {e}")
        # Fallback to a simple response if there's an error
        return f"Error loading super admin dashboard: {e}"
    

    
@app.route('/admin/queries')
@admin_required('admin')  # Allows both admin and super_admin
def admin_queries():
    """Display all queries for admin review"""
    try:
        with get_db_cursor() as cursor:
            # Get all queries with farmer information
            cursor.execute("""
                SELECT q.query_id, q.query_text, q.status, q.query_timestamp, 
                       f.phone_number, f.location 
                FROM query q 
                JOIN farmer f ON q.farmer_id = f.farmer_id 
                ORDER BY q.query_timestamp DESC
            """)
            queries = cursor.fetchall()
            
            # Get query statistics
            cursor.execute("SELECT status, COUNT(*) as count FROM query GROUP BY status")
            status_counts = cursor.fetchall()
            
            # Convert to a dictionary for easier access
            status_dict = {}
            for item in status_counts:
                status_dict[item['status']] = item['count']
            
            statistics = {
                'total_queries': len(queries),
                'status_counts': status_dict
            }
            
            return render_template('admin_queries.html', queries=queries, statistics=statistics)
            
    except Exception as e:
        logger.error(f"Error loading admin queries: {e}")
        return f"Error loading queries: {e}"
    
@app.route('/admin/knowledge-base')
@admin_required('admin')  # Allows both admin and super_admin
def admin_knowledge_base():
    """Display and manage the knowledge base for admins"""
    try:
        with get_db_cursor() as cursor:
            # First, check if the created_at column exists
            cursor.execute("SHOW COLUMNS FROM knowledge_article LIKE 'created_at'")
            has_created_at = cursor.fetchone() is not None
            
            # Build the query based on available columns
            if has_created_at:
                cursor.execute("""
                    SELECT article_id, title, content, keywords, category, 
                           created_at, updated_at 
                    FROM knowledge_article 
                    ORDER BY created_at DESC
                """)
            else:
                cursor.execute("""
                    SELECT article_id, title, content, keywords, category
                    FROM knowledge_article 
                    ORDER BY article_id DESC
                """)
                
            articles = cursor.fetchall()
            
            # Get article statistics
            cursor.execute("SELECT category, COUNT(*) as count FROM knowledge_article GROUP BY category")
            category_counts = cursor.fetchall()
            
            # Convert to a dictionary for easier access
            category_dict = {}
            for item in category_counts:
                category_dict[item['category']] = item['count']
            
            statistics = {
                'total_articles': len(articles),
                'category_counts': category_dict,
                'has_timestamps': has_created_at
            }
            
            return render_template('admin_knowledge_base.html', 
                                 articles=articles, 
                                 statistics=statistics)
            
    except Exception as e:
        logger.error(f"Error loading admin knowledge base: {e}")
        return f"Error loading knowledge base: {e}"
    
@app.route('/admin/experts')
@admin_required('super_admin')
def admin_experts():
    """Display and manage agricultural experts"""
    try:
        with get_db_cursor() as cursor:
            # Check if the is_active column exists in the agricultural_expert table
            cursor.execute("SHOW COLUMNS FROM agricultural_expert LIKE 'is_active'")
            has_is_active = cursor.fetchone() is not None

            # Fetch experts
            if has_is_active:
                cursor.execute("SELECT expert_id, expert_name, contact_info, is_active FROM agricultural_expert")
            else:
                cursor.execute("SELECT expert_id, expert_name, contact_info FROM agricultural_expert")
                
            experts = cursor.fetchall()
            
            # Convert is_active to boolean for easier handling in template if the column exists
            if has_is_active:
                for expert in experts:
                    expert['is_active'] = bool(expert['is_active'])
            
            return render_template('admin_experts.html', experts=experts, has_is_active=has_is_active)
            
    except Exception as e:
        logger.error(f"Error loading admin experts: {e}")
        return f"Error loading experts: {e}", 500
    
@app.route('/admin/settings')
@admin_required('super_admin')
def admin_settings():
    """System settings management page for super admins"""
    try:
        # Get current system settings (you would typically store these in a database)
        # For now, we'll use some default values
        system_settings = {
            'sms_enabled': True,
            'ai_threshold': 0.6,
            'auto_escalate': True,
            'max_sms_length': 160,
            'response_timeout': 24,  # hours
            'default_language': 'auto',
            'maintenance_mode': False
        }
        
        # Get Twilio configuration status
        twilio_configured = bool(os.getenv('TWILIO_ACCOUNT_SID') and os.getenv('TWILIO_AUTH_TOKEN'))
        
        # Get database status
        db_configured = bool(os.getenv('DB_HOST') and os.getenv('DB_USER') and os.getenv('DB_NAME'))
        
        return render_template('admin_settings.html', 
                             settings=system_settings,
                             twilio_configured=twilio_configured,
                             db_configured=db_configured)
            
    except Exception as e:
        logger.error(f"Error loading admin settings: {e}")
        return "Error loading admin settings", 500
    
@app.route('/admin/settings/save', methods=['POST'])
@admin_required('super_admin')
def save_settings():
    """Save system settings"""
    try:
        #  success message
        return jsonify({'success': True, 'message': 'Settings saved successfully'})
        
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return jsonify({'success': False, 'message': 'Error saving settings'}), 500
    

# Admin login route
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if 'admin_id' in session:
        # Redirect based on role
        if session.get('admin_role') == 'super_admin':
            return redirect(url_for('super_admin_dashboard'))
        else:
            return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        try:
            with get_db_cursor() as cursor:
                # Get admin from database
                cursor.execute(
                    "SELECT id, username, password_hash, full_name, role, is_active, registration_status FROM admin_users WHERE username = %s", 
                    (username,)
                )
                admin = cursor.fetchone()
                
                if admin:
                    # Special handling for super admin
                    if admin['role'] == 'super_admin':
                        # For super admin, bypass registration status check
                        if not admin['is_active']:
                            return render_template('admin_login.html', error="Super admin account is deactivated")
                        
                        # Verify password
                        if check_password_hash(admin['password_hash'], password):
                            # Update last login timestamp
                            cursor.execute(
                                "UPDATE admin_users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                                (admin['id'],)
                            )
                            
                            # Set session variables
                            session['admin_id'] = admin['id']
                            session['admin_username'] = admin['username']
                            session['admin_full_name'] = admin['full_name']
                            session['admin_role'] = admin['role']
                            
                            return redirect(url_for('super_admin_dashboard'))
                        else:
                            return render_template('admin_login.html', error="Invalid credentials")
                    
                    # Regular admin handling
                    else:
                        # Check if account is approved
                        if admin['registration_status'] != 'approved':
                            return render_template('admin_login.html', error="Account is pending approval by a super admin.")
                        
                        # Check if account is active
                        if not admin['is_active']:
                            return render_template('admin_login.html', error="Account is deactivated")
                        
                        # Verify password
                        if check_password_hash(admin['password_hash'], password):
                            # Update last login timestamp
                            cursor.execute(
                                "UPDATE admin_users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                                (admin['id'],)
                            )
                            
                            # Set session variables
                            session['admin_id'] = admin['id']
                            session['admin_username'] = admin['username']
                            session['admin_full_name'] = admin['full_name']
                            session['admin_role'] = admin['role']
                            
                            return redirect(url_for('admin_dashboard'))
                        else:
                            return render_template('admin_login.html', error="Invalid credentials")
                else:
                    return render_template('admin_login.html', error="Invalid credentials")
                    
        except Exception as e:
            logger.error(f"Error during admin login: {e}")
            return render_template('admin_login.html', error="Login error. Please try again.")
    
    return render_template('admin_login.html')

# Admin dashboard route
@app.route('/admin/dashboard')
def admin_dashboard():
    # Simple authentication check
    if 'admin_username' not in session:
        return redirect(url_for('admin_login'))
    
    try:
        with get_db_cursor() as cursor:
            # Get total queries
            cursor.execute("SELECT COUNT(*) as total FROM query")
            total_queries = cursor.fetchone()['total']
            
            # Get queries by status
            cursor.execute("SELECT status, COUNT(*) as count FROM query GROUP BY status")
            status_counts = cursor.fetchall()
            
            # Convert to a simple dictionary
            status_dict = {}
            for item in status_counts:
                status_dict[item['status']] = item['count']
            
            # Get recent queries
            cursor.execute("""
                SELECT query_id, query_text, status, query_timestamp 
                FROM query 
                ORDER BY query_timestamp DESC 
                LIMIT 10
            """)
            recent_queries = cursor.fetchall()

    
            # Prepare simple statistics
            statistics = {
                'total_queries': total_queries,
                'ai_answered': status_dict.get('answered_ai', 0),
                'escalated': status_dict.get('escalated', 0),
                'expert_answered': status_dict.get('answered_expert', 0),
                'recent_queries': recent_queries
            }

            # Add current time for the footer
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            #redirect to appropriate dashboard based on role
            if session.get('admin_role') == 'super_admin':
                return redirect(url_for('super_admin_dashboard'))
            else:
                return render_template('admin_dashboard.html', statistics=statistics, current_time=current_time)
            
            
    except Exception as e:
        logger.error(f"Error loading admin dashboard: {e}")
        return f"Error loading admin dashboard: {e}"

# Admin route to view all admin users
@app.route('/admin/users')
@admin_required('super_admin')
def admin_users():
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT id, username, full_name, email, role, is_active, created_at, last_login FROM admin_users")
            users = cursor.fetchall()
            
            return render_template('admin_users.html', users=users)
            
    except Exception as e:
        logger.error(f"Error fetching admin users: {e}")
        return "Error fetching admin users", 500

# Admin route to add a new admin user
@app.route('/admin/add-user', methods=['GET', 'POST'])
@admin_required('super_admin')
def add_admin_user():
    if request.method == 'POST':
        username = request.form.get('username')
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        role = request.form.get('role')
        password = request.form.get('password')
        is_active = 1 if request.form.get('is_active') else 0
        
        if not all([username, full_name, role, password]):
            return render_template('add_admin_user.html', error="Required fields are missing")
        
        try:
            # Hash the password
            hashed_password = generate_password_hash(password)
            
            with get_db_cursor() as cursor:
                user_id = f"ADM_{uuid.uuid4().hex[:8]}"
                
                cursor.execute(
                    "INSERT INTO admin_users (id, username, full_name, email, role, password_hash, is_active) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (user_id, username, full_name, email, role, hashed_password, is_active)
                )
                
                return redirect(url_for('admin_users'))
                
        except mysql.connector.IntegrityError:
            return render_template('add_admin_user.html', error="Username already exists")
        except Exception as e:
            logger.error(f"Error adding admin user: {e}")
            return render_template('add_admin_user.html', error="Error adding admin user")
    
    return render_template('add_admin_user.html')

# Admin route to toggle user status
@app.route('/admin/toggle-user-status/<user_id>')
@admin_required('super_admin')
def toggle_user_status(user_id):
    try:
        with get_db_cursor() as cursor:
            # Get current status
            cursor.execute("SELECT is_active FROM admin_users WHERE id = %s", (user_id,))
            user = cursor.fetchone()
            
            if user:
                new_status = 0 if user['is_active'] else 1
                
                cursor.execute(
                    "UPDATE admin_users SET is_active = %s WHERE id = %s",
                    (new_status, user_id)
                )
                
                return redirect(url_for('admin_users'))
            else:
                return "User not found", 404
                
    except Exception as e:
        logger.error(f"Error toggling user status: {e}")
        return "Error toggling user status", 500

# Admin route to delete a user
@app.route('/admin/delete-user/<user_id>')
@admin_required('super_admin')
def delete_admin_user(user_id):
    try:
        # Prevent self-deletion
        if user_id == session.get('admin_id'):
            return "You cannot delete your own account", 400
            
        with get_db_cursor() as cursor:
            cursor.execute("DELETE FROM admin_users WHERE id = %s", (user_id,))
            
            return redirect(url_for('admin_users'))
            
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return "Error deleting user", 500
    
# Route to view pending admin approvals
@app.route('/admin/approvals')
@admin_required('super_admin')
def admin_approvals():
    """Display pending admin approvals"""
    try:
        with get_db_cursor() as cursor:
            # Check if registration_status column exists in admin_users
            cursor.execute("SHOW COLUMNS FROM admin_users LIKE 'registration_status'")
            has_registration_status = cursor.fetchone() is not None
            
            # Get pending admin registrations
            if has_registration_status:
                cursor.execute("""
                    SELECT id, username, full_name, email, registration_date 
                    FROM admin_users 
                    WHERE registration_status = 'pending' AND role = 'admin'
                    ORDER BY registration_date DESC
                """)
            else:
                # Fallback if registration_status column doesn't exist
                cursor.execute("""
                    SELECT id, username, full_name, email, created_at as registration_date 
                    FROM admin_users 
                    WHERE is_active = 0 AND role = 'admin'
                    ORDER BY created_at DESC
                """)
            pending_admins = cursor.fetchall()
            
            # Get approved admins
            if has_registration_status:
                cursor.execute("""
                    SELECT id, username, full_name, email, is_active, approved_by, approval_date 
                    FROM admin_users 
                    WHERE registration_status = 'approved' AND role = 'admin'
                    ORDER BY approval_date DESC
                """)
            else:
                cursor.execute("""
                    SELECT id, username, full_name, email, is_active, 'N/A' as approved_by, created_at as approval_date 
                    FROM admin_users 
                    WHERE is_active = 1 AND role = 'admin'
                    ORDER BY created_at DESC
                """)
            approved_admins = cursor.fetchall()
            
            return render_template('admin_approvals.html', 
                                 pending_admins=pending_admins,
                                 approved_admins=approved_admins,
                                 has_registration_status=has_registration_status)
            
    except Exception as e:
        logger.error(f"Error fetching admin approvals: {e}")
        logger.error(traceback.format_exc())
        return "Error fetching admin approvals", 500

# Route to approve an admin
@app.route('/admin/approve/admin/<admin_id>')
@admin_required('super_admin')
def approve_admin(admin_id):
    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                "UPDATE admin_users SET is_active = 1, registration_status = 'approved', approved_by = %s, approval_date = CURRENT_TIMESTAMP WHERE id = %s",
                (session.get('admin_id'), admin_id)
            )
            flash('Admin account approved successfully!', 'success')
            return redirect(url_for('admin_approvals'))
            
    except Exception as e:
        logger.error(f"Error approving admin: {e}")
        flash('Error approving admin account.', 'error')
        return redirect(url_for('admin_approvals'))

# Route to reject an admin
@app.route('/admin/reject/admin/<admin_id>')
@admin_required('super_admin')
def reject_admin(admin_id):
    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                "UPDATE admin_users SET registration_status = 'rejected', approved_by = %s, approval_date = CURRENT_TIMESTAMP WHERE id = %s",
                (session.get('admin_id'), admin_id)
            )
            flash('Admin account rejected.', 'success')
            return redirect(url_for('admin_approvals'))
            
    except Exception as e:
        logger.error(f"Error rejecting admin: {e}")
        flash('Error rejecting admin account.', 'error')
        return redirect(url_for('admin_approvals'))
    
# Admin logout route
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    session.pop('admin_username', None)
    session.pop('admin_full_name', None)
    session.pop('admin_role', None)
    return redirect(url_for('home'))

# Test SMS endpoint
@app.route('/test-sms', methods=['GET'])
def test_sms():
    """Test endpoint to verify Twilio SMS functionality"""
    test_number = os.getenv('TEST_PHONE_NUMBER')
    test_message = "Test message from Agricultural Chatbot"
    
    if not test_number:
        return "TEST_PHONE_NUMBER not set in environment variables", 500
    
    if not twilio_client:
        return "Twilio client not initialized", 500
    
    try:
        message = twilio_client.messages.create(
            body=test_message,
            from_=os.getenv('TWILIO_PHONE_NUMBER'),
            to=test_number
        )
        return f"Test message sent! SID: {message.sid}, Status: {message.status}"
    except Exception as e:
        return f"Failed to send test message: {str(e)}", 500

# Language test endpoint
@app.route('/api/test-language', methods=['POST'])
def test_language():
    """API endpoint to test language detection"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    query_text = data['query']
    try:
        language = nlp_processor.detect_language(query_text)
        english_translation = nlp_processor.translate_to_english(query_text) if language == 'sw' else query_text
        swahili_translation = nlp_processor.translate_to_swahili(query_text) if language == 'en' else query_text
        
        return jsonify({
            'query': query_text, 
            'detected_language': language,
            'english_translation': english_translation,
            'swahili_translation': swahili_translation
        })
    except Exception as e:
        logger.error(f"Error in language test: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)