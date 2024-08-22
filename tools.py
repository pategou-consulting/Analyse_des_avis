import pandas as pd
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import nltk
import emoji

path="C:/Users/batom/models/facebook_bart_large_mnli"
nli_model = AutoModelForSequenceClassification.from_pretrained(path)    
tokenizer = AutoTokenizer.from_pretrained(path)
classifier = pipeline("zero-shot-classification", model=nli_model, tokenizer=tokenizer)

# Load the NLTK sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Possible labels
candidate_labels = ["positif", "négatif", "neutre"]

# Function to analyze reviews with BART
def analyze_review_bart(review):
    result = classifier(review, candidate_labels)
    return result['labels'][0], result['scores'][0]

# Function for sentiment analysis with NLTK considering emojis
def analyze_sentiment(review):
    review = emoji.demojize(review)  # Convert emojis to text
    scores = sia.polarity_scores(review)
    return scores['compound']

# Function to preprocess reviews (including emojis)
def preprocess(text):
    text = emoji.demojize(text)  # Convert emojis to text
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('french'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

# Example review data
data = {
    "review_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "review_text": [
        "La qualité du produit est excellente, je suis très satisfait 😊.",
        "La livraison a été très lente 😠, mais le produit est bon 👍.",
        "Le service client est terrible 😡 et le produit est défectueux 😤.",
        "Excellent service ! J'ai commandé un produit et il est arrivé bien avant la date prévue. Le suivi de commande était parfait et l'emballage soigné. Je suis vraiment satisfait de mon achat et je recommanderai sans hésiter cette marketplace à mes amis !",
        "Très bonne expérience d'achat. Le site est facile à naviguer, et j'ai trouvé exactement ce que je cherchais. Le produit est conforme à la description et la qualité est au rendez-vous. Je reviendrai pour mes prochains achats.",
        "Extrêmement déçu par cette commande. Le produit que j'ai reçu était endommagé et la qualité laissait vraiment à désirer. De plus, le service client a mis beaucoup de temps à répondre à mes plaintes. Je ne recommande pas cette marketplace.",
        "Service après-vente inexistant. J'ai eu un problème avec ma commande, mais impossible de contacter qui que ce soit pour résoudre mon problème. C'est dommage, car la marketplace semblait prometteuse au début.",
        "Expérience d'achat correcte, sans plus. Le produit est arrivé à temps et correspondait à la description. Toutefois, l'emballage aurait pu être un peu plus soigné. Pas de gros soucis, mais pas d'enthousiasme non plus.",
        "Rien de particulier à signaler. La commande est arrivée dans les délais, mais la qualité du produit est juste moyenne. C'est acceptable pour le prix, mais je ne suis ni particulièrement satisfait ni insatisfait."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)
# Apply preprocessing
df['tokens'] = df['review_text'].apply(preprocess)

# Create a dictionary and corpus
dictionary = Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]

# Train the LDA model
lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

# Function to extract topics
def get_topics(review):
    bow = dictionary.doc2bow(preprocess(review))
    topics = lda_model.get_document_topics(bow)
    topics = sorted(topics, key=lambda x: x[1], reverse=True)
    return topics[0][0] if topics else None