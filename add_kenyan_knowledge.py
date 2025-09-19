import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    conn = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )
    return conn

def add_kenyan_agricultural_knowledge():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Kenyan-specific agricultural knowledge with regions
    kenyan_knowledge = [
        # Pest control in Kenya
        ('ART_105', 'Fall Armyworm Control in Kenya', 
         'For fall armyworm in maize, use recommended pesticides like Escort, Rocket, or Belt. Apply when worms are small for best results. Practice crop rotation and field sanitation.',
         'Kudhibiti kiwavi kwenye mahindi, tumia dawa zilizopendekezwa kama Escort, Rocket, au Belt. Tumia wakati funza ni ndogo kwa matokeo bora. Fanya mzunguko wa mazao na usafisha shamba.',
         'fall armyworm, maize, kenya, pesticide, escort, rocket, belt, crop rotation', 'pest', 'Rift Valley, Western, Nyanza'),
        
        # Soil management for Kenyan conditions
        ('ART_106', 'Soil Management in Kenya', 
         'Test your soil at KALRO or agricultural extension offices. For acidic soils common in Kenya, apply lime. For nutrient-deficient soils, use organic manure and appropriate fertilizers.',
         'Pima udongo wako KALRO au ofisi za ugatuzi wa kilimo. Kwa udongo wenye asidi uliokithiri Kenya, tumia chokaa. Kwa udongo duni wa virutubisho, tumia samadi ya kikaboni na mbolea stahiki.',
         'soil management, kenya, kalro, acidic soil, lime, organic manure, fertilizer', 'soil', 'Central, Eastern, Rift Valley'),
        
        # Kenyan weather patterns
        ('ART_107', 'Kenyan Weather Patterns', 
         'Kenya has two main rainy seasons: Long rains (March-May) and short rains (October-December). Plan planting accordingly. Use Kenya Meteorological Department forecasts for accurate information.',
         'Kenya ina misimu miwili kuu ya mvua: Mvua za masika (Machi-Mei) na mvua za vuli (Oktoba-Desemba). Panga upandaji ipasavyo. Tumia utabiri wa Idara ya Hali ya Hewa ya Kenya kwa taarifa sahihi.',
         'kenya weather, rainy seasons, long rains, short rains, meteorological department', 'weather', 'National'),
        
        # Common Kenyan crops - Beans
        ('ART_108', 'Growing Beans in Kenya', 
         'Beans do well in most parts of Kenya. Plant at spacing of 50cm between rows and 15cm within rows. Use certified seeds from KALRO or approved stockists.',
         'Maharagwe hukua vizuri katika sehemu nyingi za Kenya. Panda kwa umbali wa sm 50 kati ya mistari na sm 15 ndani ya mstari. Tumia mbegu zilizosajiliwa kutoka KALRO au wauzaji walioidhinishwa.',
         'beans, kenya, spacing, kalro, certified seeds', 'crop', 'Central, Western, Rift Valley'),
        
        # Common Kenyan crops - Maize
        ('ART_110', 'Maize Farming in Kenya', 
         'Maize is Kenya staple food. Plant at onset of rains. Use recommended spacing of 75cm between rows and 30cm between plants. Apply DAP fertilizer at planting and CAN after 3-4 weeks.',
         'Mahindi ni chakula kikuu cha Kenya. Panda mwanzo wa mvua. Tumia umbali unaopendekezwa wa sm 75 kati ya mistari na sm 30 kati ya mimea. Tumia mbolea ya DAP wakati wa kupanda na CAN baada ya wiki 3-4.',
         'maize, kenya, spacing, dap, can, fertilizer', 'crop', 'Rift Valley, Western, Nyanza'),
        
        # Tea growing regions
        ('ART_111', 'Tea Growing in Kenya', 
         'Tea grows best in high altitude areas with well-distributed rainfall. Prune regularly and pluck two leaves and a bud for quality tea. Use recommended fertilizers for tea.',
         'Chai hukua vizuri katika maeneo ya milima yenye mvua iliyosambazwa vizuri. Puna mara kwa mara na kuvuna majani mawili na chipukizi moja kwa chai bora. Tumia mbolea zilizopendekezwa kwa chai.',
         'tea, kenya, high altitude, pruning, fertilizer', 'crop', 'Central, Rift Valley, Western'),
        
        # Coffee growing regions
        ('ART_112', 'Coffee Farming in Kenya', 
         'Coffee requires well-drained soils in high altitude areas. Practice mulching to conserve moisture. Control coffee berry disease with recommended fungicides.',
         'Kahawa inahitaji udongo wenye mitiririko mzuri katika maeneo ya milima. Fanya malisho ya kuhifadhi unyevu. Kudhibiti ugonjwa wa beri ya kahawa kwa fungicides zilizopendekezwa.',
         'coffee, kenya, high altitude, mulching, berry disease', 'crop', 'Central, Eastern, Rift Valley'),
        
        # Market information
        ('ART_109', 'Agricultural Markets in Kenya', 
         'Check NCPB for current cereal prices. For vegetables, visit local markets or use platforms like M-Farm for price information. Consider contract farming for stable markets.',
         'Angalia NCPB kwa bei za sasa za nafaka. Kwa mboga, tembelea soko la karibu au tumia majukwaa kama M-Farm kwa taarifa za bei. Fikiria kulima kwa mkataba kwa masoko thabiti.',
         'agricultural markets, kenya, ncpb, m-farm, contract farming', 'market', 'National'),
        
        # Dairy farming regions
        ('ART_113', 'Dairy Farming in Kenya', 
         'For high milk production, feed dairy cows on quality fodder and concentrates. Practice zero-grazing in areas with limited land. Ensure regular vaccination and deworming.',
         'Kwa uzalishaji mwingi wa maziwa, kulisha ngombe wa maziwa kwa malisho bora na concentrates. Fanya ukulima wa sifuri katika maeneo yenye ardhi ndogo. Hakikisha chanjo mara kwa mara na kuondoa minyoo.',
         'dairy, kenya, zero-grazing, fodder, vaccination', 'livestock', 'Rift Valley, Central, Western'),
        
        # Arid and semi-arid regions agriculture
        ('ART_114', 'Farming in Arid Areas', 
         'In arid areas, practice water harvesting and drought-resistant crops like sorghum, millet, and drought-tolerant beans. Use zai pits and mulching to conserve moisture.',
         'Katika maeneo yangi, fanya uvunaji wa maji na mazao yanayostahimili ukame kama mtama, ulezi na maharagwe yanayostahimili ukame. Tumia mashimo ya zai na malisho ya kuhifadhi unyevu.',
         'arid, semi-arid, kenya, water harvesting, drought-resistant crops', 'general', 'Eastern, North Eastern, Coast'),
    ]
    
    try:
        for article in kenyan_knowledge:
            article_id, title, content, content_sw, keywords, category, region = article
            # Check if article already exists
            cursor.execute("SELECT article_id FROM knowledge_article WHERE article_id = %s", (article_id,))
            if cursor.fetchone():
                print(f"Article {article_id} already exists. Skipping.")
            else:
                cursor.execute(
                    "INSERT INTO knowledge_article (article_id, title, content, content_sw, keywords, category, region) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (article_id, title, content, content_sw, keywords, category, region)
                )
                print(f"Added article {article_id} for region(s): {region}")
        conn.commit()
        print("Kenyan agricultural knowledge with regions added successfully!")
        
    except Exception as e:
        print(f"Error adding knowledge: {e}")
        conn.rollback()
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    add_kenyan_agricultural_knowledge()