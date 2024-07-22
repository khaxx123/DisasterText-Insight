from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, MilvusException
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.corpus import wordnet
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

nltk.download('wordnet')

# Connect to Milvus
try:
    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus")
except MilvusException as e:
    print(f"Failed to connect to Milvus: {e}")
    exit(1)

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),  # Assuming 384-dim vectors for SentenceTransformer
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)  # Add text field to store sentences
]
schema = CollectionSchema(fields, description="Text similarity collection")

# Create collection
collection_name = "text_similarity_collection"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
collection = Collection(name=collection_name, schema=schema)
print(f"Collection '{collection_name}' created.")

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Diverse sentences

sentences = [
    "Severe earthquake hits coastal city causing massive damage.",
    "Flooding devastates rural communities after heavy rains.",
    "Wildfire spreads rapidly, forcing thousands to evacuate.",
    "Hurricane brings strong winds and torrential rain to the region.",
    "Landslide buries homes in mountainous area, dozens missing.",
    "Tornado destroys small town, leaving residents homeless.",
    "Volcanic eruption leads to widespread ashfall and lava flows.",
    "Tsunami warning issued after powerful offshore earthquake.",
    "Heatwave causes record temperatures and health concerns.",
    "Blizzard blankets city in snow, disrupting travel and power.",
    "Severe drought affects agricultural production and water supply.",
    "Typhoon wreaks havoc on coastal communities with high winds.",
    "Avalanche traps hikers in remote mountainous region.",
    "Sandstorm reduces visibility and causes respiratory issues.",
    "Mudslide damages infrastructure and homes in affected areas.",
    "Cyclone brings heavy rain and flooding to island nation.",
    "Forest fire threatens wildlife and natural habitats.",
    "Ice storm leads to widespread power outages and accidents.",
    "Severe thunderstorm causes flash flooding in urban areas.",
    "Chemical spill contaminates river, impacting local water supply.",
    "Earthquake aftershocks continue to rattle already damaged region.",
    "Floodwaters rise rapidly, overwhelming levees and dams.",
    "Volcanic ash cloud disrupts air travel and poses health risks.",
    "Tornado outbreak hits multiple states, causing widespread destruction.",
    "Landslide blocks major highway, cutting off access to towns.",
    "Hurricane storm surge inundates coastal neighborhoods.",
    "Wildfire smoke leads to air quality warnings and health advisories.",
    "Tsunami wave destroys coastal infrastructure and homes.",
    "Heatwave causes widespread power outages and health emergencies.",
    "Blizzard conditions lead to road closures and stranded travelers.",
    "Drought leads to water rationing and agricultural losses.",
    "Typhoon triggers landslides and flooding in mountainous areas.",
    "Avalanche rescue efforts underway to locate missing individuals.",
    "Sandstorm disrupts transportation and outdoor activities.",
    "Mudslide buries vehicles and homes, complicating rescue efforts.",
    "Cyclone winds cause extensive damage to buildings and infrastructure.",
    "Forest fire containment efforts hampered by strong winds.",
    "Ice storm damages power lines and tree limbs, causing outages.",
    "Severe thunderstorm lightning strikes ignite multiple fires.",
    "Chemical spill prompts evacuation of nearby residents.",
    "Earthquake causes buildings to collapse, trapping people inside.",
    "Floodwaters breach levees, leading to emergency evacuations.",
    "Volcanic eruption forces evacuation of nearby communities.",
    "Tornado leaves a path of destruction through several towns.",
    "Landslide debris blocks river, creating potential flood risk.",
    "Hurricane winds topple trees and power lines, causing outages.",
    "Wildfire threatens homes and prompts mandatory evacuations.",
    "Tsunami alert issued after underwater volcanic eruption.",
    "Heatwave strains power grid and leads to rolling blackouts.",
    "Blizzard conditions cause multiple vehicle accidents on highways.",
    "Drought impacts wildlife and leads to water restrictions.",
    "Typhoon causes widespread flooding and landslides in affected areas.",
    "Avalanche warnings issued for high-risk mountain regions.",
    "Sandstorm causes visibility issues and flight cancellations.",
    "Mudslide damages roads and disrupts transportation.",
    "Cyclone brings heavy rainfall and coastal erosion.",
    "Forest fire spreads rapidly due to dry conditions and high winds.",
    "Ice storm creates hazardous driving conditions and power outages.",
    "Severe thunderstorm causes hail damage to crops and property.",
    "Chemical spill response teams work to contain contamination.",
    "Earthquake damages infrastructure and disrupts daily life.",
    "Floodwaters force residents to seek shelter in emergency centers.",
    "Volcanic eruption spews ash and lava, affecting nearby areas.",
    "Tornado emergency declared as multiple tornadoes touch down.",
    "Landslide risk increases after days of heavy rainfall.",
    "Hurricane evacuation orders issued for coastal regions.",
    "Wildfire containment efforts continue as fire spreads.",
    "Tsunami evacuation drills conducted in coastal communities.",
    "Heatwave advisory issued, urging residents to stay hydrated.",
    "Blizzard leads to closures of schools and businesses.",
    "Drought leads to reduced crop yields and higher food prices.",
    "Typhoon causes widespread power outages and infrastructure damage.",
    "Avalanche rescue teams search for missing climbers.",
    "Sandstorm warnings issued for desert regions.",
    "Mudslide recovery efforts underway to clear debris.",
    "Cyclone impacts lead to widespread flooding and damage.",
    "Forest fire destroys hundreds of acres of woodland.",
    "Ice storm causes transportation disruptions and accidents.",
    "Severe thunderstorm leads to flash flooding in low-lying areas.",
    "Chemical spill cleanup efforts continue to prevent environmental damage.",
    "Earthquake rattles region, causing minor injuries and damage.",
    "Floodwaters recede, leaving behind extensive damage and debris.",
    "Volcanic eruption disrupts air travel and poses health risks.",
    "Tornado recovery efforts focus on rebuilding and aid distribution.",
    "Landslide warnings remain in effect for at-risk areas.",
    "Hurricane aftermath includes power outages and infrastructure damage.",
    "Wildfire smoke spreads across multiple states, affecting air quality.",
    "Tsunami waves reach shore, causing significant damage.",
    "Heatwave continues, putting vulnerable populations at risk.",
    "Blizzard conditions persist, complicating travel and emergency response.",
    "Drought leads to increased wildfire risk and water scarcity.",
    "Typhoon recovery efforts focus on restoring power and infrastructure.",
    "Avalanche danger remains high in certain mountain regions.",
    "Sandstorm impacts travel and daily life in affected areas.",
    "Mudslide cleanup efforts progress slowly due to challenging conditions.",
    "Cyclone warnings remain in effect as storm approaches.",
    "Forest fire containment strategies focus on protecting communities.",
    "Ice storm recovery efforts prioritize restoring power and clearing roads.",
    "Severe thunderstorm causes widespread damage and power outages.",
    "Chemical spill investigation underway to determine cause and impact.",
    "Earthquake leaves many homeless and in need of assistance.",
    "Floodwaters subside, revealing the extent of the destruction.",
    "Volcanic eruption leads to evacuation of nearby towns.",
    "Tornado outbreak leaves a trail of devastation across multiple states.",
    "Landslide blocks access to key infrastructure and facilities.",
    "Hurricane recovery efforts focus on clearing debris and restoring services.",
    "Wildfire spreads to residential areas, threatening homes.",
    "Tsunami alert lifted as threat subsides.",
    "Heatwave prompts health advisories and emergency measures.",
    "Blizzard leads to delays and cancellations of flights and trains.",
    "Drought continues to affect water supply and agricultural output.",
    "Typhoon cleanup efforts focus on clearing debris and restoring services.",
    "Avalanche risk remains high in certain regions.",
    "Sandstorm causes disruptions to travel and outdoor activities.",
    "Mudslide recovery efforts focus on clearing major roadways.",
    "Cyclone brings heavy rainfall and strong winds to coastal areas.",
    "Forest fire burns thousands of acres, threatening nearby towns.",
    "Ice storm causes widespread power outages and transportation disruptions.",
    "Severe thunderstorm leads to flooding and property damage.",
    "Chemical spill prompts investigation and environmental monitoring.",
    "Earthquake recovery efforts focus on rebuilding and aid distribution.",
    "Floodwaters recede, leaving behind extensive damage and debris.",
    "Volcanic eruption leads to evacuation and air quality concerns.",
    "Tornado outbreak causes widespread destruction across multiple states.",
    "Landslide blocks major highways and disrupts transportation.",
    "Hurricane recovery efforts focus on clearing debris and restoring power.",
    "Wildfire smoke spreads across large areas, affecting air quality.",
    "Tsunami waves cause significant damage to coastal communities.",
    "Heatwave continues, posing health risks to vulnerable populations.",
    "Blizzard conditions lead to road closures and travel disruptions."
]



# Compute embeddings
embeddings = model.encode(sentences)

# Insert data into Milvus
ids = [i for i in range(len(sentences))]
data = [ids, embeddings.tolist(), sentences]
collection.insert(data)
print(f"Inserted {len(sentences)} texts into collection '{collection_name}'.")

# Create an index for the collection
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 64},
    "metric_type": "L2"
}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()
print("Index created and collection loaded into memory.")

# Function to get synonyms using WordNet
def get_related_words(keyword):
    synonyms = set()
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

# Function to perform similarity search
def search_similar_texts(query_text, threshold=2.0):
    query_embedding = model.encode([query_text])[0]
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "vector", search_params, limit=10, output_fields=["id", "text"])

    # Filter results based on the distance threshold
    filtered_results = [hit for hit in results[0] if hit.distance <= threshold]
    return filtered_results

# Loop to handle multiple test cases
while True:
    # Get user input for the keyword
    keyword = input("Enter the keyword to filter results (e.g., 'sport'), or type 'exit' to quit: ").strip().lower()
    if keyword == 'exit':
        break

    # Get related words
    related_words = get_related_words(keyword)
    related_words.add(keyword)  # Include the original keyword

    # Perform a similarity search for the user's keyword and related words
    all_results = []
    for word in related_words:
        query_text = word
        results = search_similar_texts(query_text)
        all_results.extend(results)

    # Remove duplicates and ensure unique results
    unique_results = {}
    for hit in all_results:
        if hit.id not in unique_results or hit.distance < unique_results[hit.id].distance:
            unique_results[hit.id] = hit

    # Sort results by distance
    sorted_results = sorted(unique_results.values(), key=lambda x: x.distance)

    # Print only the top 3 results with ranking
    print("Search results after filtering:")
    for idx, hit in enumerate(sorted_results[:3]):
        rank = ["first", "second", "third"][idx]
        print(f"Rank: {rank}, ID: {hit.id}, Distance: {hit.distance:.4f}, Sentence: {hit.entity.get('text')}")

# Disconnect from Milvus
connections.disconnect("default")
print("Disconnected from Milvus")

