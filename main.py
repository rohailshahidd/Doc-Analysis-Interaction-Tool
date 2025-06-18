import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader, WikipediaLoader, HNLoader, DirectoryLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2TokenizerFast, pipeline, BartForConditionalGeneration, BartTokenizer
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.chains import ConversationChain, RetrievalQA
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from pyvis.network import Network
import tiktoken
import time
import spacy
import pytesseract
from PIL import Image
import speech_recognition as sr
import pyttsx3
import re
import nltk
from nltk.corpus import wordnet
from gensim.summarization import keywords
from textblob import TextBlob
from wordcloud import WordCloud
import yake
import pycountry
import geocoder
import folium
from googletrans import Translator
import cv2
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
from pydub import AudioSegment
import pygame
from gtts import gTTS

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

load_dotenv()

openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_env_key = os.environ["PINECONE_ENV_KEY"]

embeddings = OpenAIEmbeddings()
chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
memory = ConversationSummaryBufferMemory(llm=chatgpt, max_token_limit=100)
conversation = ConversationChain(llm=chatgpt, memory=memory, verbose=True)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = Translator()

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=count_tokens
)

def load_and_process_data(source_type, source_path):
    if source_type == "wikipedia":
        loader = WikipediaLoader(source_path)
    elif source_type == "pdf":
        loader = PyPDFLoader(source_path)
    elif source_type == "hacker_news":
        loader = HNLoader(source_path)
    elif source_type == "directory":
        loader = DirectoryLoader(source_path, glob="**/*.pdf")
    elif source_type == "youtube":
        loader = YoutubeLoader.from_youtube_url(source_path, add_video_info=True)
    elif source_type == "image":
        return [Document(page_content=pytesseract.image_to_string(Image.open(source_path)))]
    elif source_type == "audio":
        recognizer = sr.Recognizer()
        with sr.AudioFile(source_path) as source:
            audio = recognizer.record(source)
        return [Document(page_content=recognizer.recognize_google(audio))]
    elif source_type == "video":
        video = VideoFileClip(source_path)
        audio = video.audio
        audio.write_audiofile("temp_audio.wav")
        video.close()
        
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp_audio.wav") as source:
            audio = recognizer.record(source)
        
        os.remove("temp_audio.wav")
        return [Document(page_content=recognizer.recognize_google(audio))]
    else:
        raise ValueError("Unsupported source type")
    
    data = loader.load()
    chunks = text_splitter.split_documents(data)
    return chunks

def create_vector_store(chunks, store_type="faiss", index_name="langchain"):
    if store_type == "faiss":
        return FAISS.from_documents(chunks, embeddings)
    elif store_type == "pinecone":
        import pinecone
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env_key)
        return Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    else:
        raise ValueError("Unsupported vector store type")

def analyze_token_distribution(chunks):
    token_counts = [count_tokens(chunk.page_content) for chunk in chunks]
    df = pd.DataFrame({'Token Count': token_counts})
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Token Count', kde=True)
    plt.title("Token Count Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.show()

def ask_question(query, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=chatgpt,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    
    with get_openai_callback() as cb:
        start_time = time.time()
        result = qa_chain({"query": query})
        end_time = time.time()
    
    answer = result['result']
    source_docs = result['source_documents']
    
    print(f"Answer: {answer}\n")
    print("Sources:")
    for i, doc in enumerate(source_docs, 1):
        print(f"Source {i}:")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}\n")
    
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Tokens used: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost:.4f}\n")
    
    return answer, source_docs

def interactive_qa_session(db):
    print("Starting interactive Q&A session. Type 'exit' to end the session.")
    
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()
    
    while True:
        choice = input("Choose input method (text/voice): ")
        
        if choice.lower() == 'text':
            query = input("Your question: ")
        elif choice.lower() == 'voice':
            with sr.Microphone() as source:
                print("Speak your question...")
                audio = recognizer.listen(source)
            try:
                query = recognizer.recognize_google(audio)
                print(f"You asked: {query}")
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that.")
                continue
        else:
            print("Invalid choice. Please choose 'text' or 'voice'.")
            continue
        
        if query.lower() == 'exit':
            break
        
        answer, _ = ask_question(query, db)
        
        sentiment = sentiment_analyzer(answer)[0]
        print(f"Answer sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})")
        
        engine.say(answer)
        engine.runAndWait()

def summarize_document(db):
    return ask_question("Provide a comprehensive summary of the entire document.", db)

def extract_key_concepts(db, num_concepts=5):
    return ask_question(f"List and briefly explain the {num_concepts} most important concepts from the document.", db)

def compare_documents(db1, db2):
    combined_db = FAISS.merge_from([db1, db2])
    return ask_question("Compare and contrast the main ideas presented in both documents.", combined_db)

def analyze_document_structure(chunks):
    G = nx.Graph()
    for i, chunk in enumerate(chunks):
        G.add_node(i, content=chunk.page_content[:50])
        for j, other_chunk in enumerate(chunks[i+1:], i+1):
            similarity = embeddings.similarity(chunk.page_content, other_chunk.page_content)
            if similarity > 0.5:
                G.add_edge(i, j, weight=similarity)
    
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    for node in G.nodes():
        net.add_node(node, label=G.nodes[node]['content'])
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], value=edge[2]['weight'])
    
    net.show("document_structure.html")

def topic_modeling(chunks, num_topics=5):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform([chunk.page_content for chunk in chunks])
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

def extract_entities(chunks):
    entities = {}
    for chunk in chunks:
        doc = nlp(chunk.page_content)
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = set()
            entities[ent.label_].add(ent.text)
    
    for label, entity_set in entities.items():
        print(f"{label}: {', '.join(entity_set)}")

def generate_quiz(db):
    quiz_prompt = "Generate a 5-question multiple-choice quiz based on the document content. Format each question as 'Q: [question]\\nA: [option1]\\nB: [option2]\\nC: [option3]\\nD: [option4]\\nCorrect: [correct_option]'"
    quiz, _ = ask_question(quiz_prompt, db)
    
    questions = re.findall(r'Q: (.+?)(?=\nQ:|$)', quiz, re.DOTALL)
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}:")
        lines = question.strip().split('\n')
        for line in lines[:-1]:  # Exclude the "Correct:" line
            print(line)
        user_answer = input("Your answer (A/B/C/D): ").upper()
        correct_answer = lines[-1].split(': ')[1]
        if user_answer == correct_answer:
            print("Correct!")
        else:
            print(f"Incorrect. The correct answer is {correct_answer}.")
        print()

def generate_word_cloud(chunks):
    text = ' '.join([chunk.page_content for chunk in chunks])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()

def extract_keywords(chunks):
    text = ' '.join([chunk.page_content for chunk in chunks])
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    
    print("Top Keywords:")
    for kw, score in keywords[:10]:
        print(f"{kw}: {score}")

def generate_mind_map(db):
    topics_prompt = "Generate a hierarchical mind map of the main topics and subtopics in the document. Format as 'Main Topic 1:\\n- Subtopic 1.1\\n- Subtopic 1.2\\nMain Topic 2:\\n- Subtopic 2.1\\n- Subtopic 2.2'"
    mind_map, _ = ask_question(topics_prompt, db)
    
    G = nx.Graph()
    current_main_topic = None
    
    for line in mind_map.split('\n'):
        if not line.startswith('- '):
            current_main_topic = line.strip(':')
            G.add_node(current_main_topic)
        else:
            subtopic = line.strip('- ')
            G.add_node(subtopic)
            G.add_edge(current_main_topic, subtopic)
    
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    for node in G.nodes():
        net.add_node(node, label=node)
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
    net.show("mind_map.html")

def generate_timeline(db):
    timeline_prompt = "Generate a timeline of key events mentioned in the document. Format as 'Date/Time: Event'"
    timeline, _ = ask_question(timeline_prompt, db)
    
    events = timeline.split('\n')
    dates = [event.split(':')[0] for event in events]
    descriptions = [event.split(':')[1] for event in events]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(dates)
    ax.set_xlabel('Event')
    ax.set_title('Timeline of Key Events')
    
    for i, description in enumerate(descriptions):
        ax.annotate(description, (0, i), xytext=(5, 0), textcoords="offset points")
    
    plt.tight_layout()
    plt.show()

def generate_analogies(db):
    analogy_prompt = "Generate 5 creative analogies to explain complex concepts in the document."
    analogies, _ = ask_question(analogy_prompt, db)
    print(analogies)

def semantic_search(query, db):
    results = db.similarity_search(query, k=5)
    print(f"Top 5 semantically similar passages for query: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.page_content[:200]}...\n")

def generate_visual_summary(db):
    summary_prompt = "Generate a visual summary of the document using emoji and simple ASCII art."
    visual_summary, _ = ask_question(summary_prompt, db)
    print(visual_summary)

def translate_document(chunks, target_language='es'):
    translated_chunks = []
    for chunk in chunks:
        translated_text = translator.translate(chunk.page_content, dest=target_language).text
        translated_chunks.append(Document(page_content=translated_text, metadata=chunk.metadata))
    return translated_chunks
def generate_study_guide(db):
    study_guide_prompt = "Create a comprehensive study guide for the main topics in the document, including key points, definitions, and example questions."
    study_guide, _ = ask_question(study_guide_prompt, db)
    print(study_guide)

def generate_counter_arguments(db):
    counter_args_prompt = "Identify the main arguments in the document and generate possible counter-arguments or alternative viewpoints."
    counter_arguments, _ = ask_question(counter_args_prompt, db)
    print(counter_arguments)

def generate_interdisciplinary_connections(db):
    connections_prompt = "Identify potential connections between the topics in this document and other fields of study or real-world applications."
    connections, _ = ask_question(connections_prompt, db)
    print(connections)

def generate_future_scenarios(db):
    scenarios_prompt = "Based on the information in the document, generate three possible future scenarios or predictions related to the main topics."
    scenarios, _ = ask_question(scenarios_prompt, db)
    print(scenarios)

def generate_ethical_analysis(db):
    ethics_prompt = "Provide an ethical analysis of the main ideas or findings presented in the document, considering potential implications and ethical concerns."
    ethical_analysis, _ = ask_question(ethics_prompt, db)
    print(ethical_analysis) 
def interactive_storytelling(db):
    print("Welcome to the interactive storytelling session based on the document!")
    story_start, _ = ask_question("Create a short story opening based on the main themes of the document.", db)
    print(story_start)
    
    while True:
        choice = input("What would you like to happen next? (or type 'exit' to end): ")
        if choice.lower() == 'exit':
            break
        continuation, _ = ask_question(f"Continue the story based on this choice: {choice}", db)
        print(continuation)

def generate_multimedia_presentation(db):
    presentation_prompt = "Create an outline for a multimedia presentation of the document's key points, including suggestions for visuals, audio, and interactive elements."
    presentation_outline, _ = ask_question(presentation_prompt, db)
    print(presentation_outline)

def generate_debate_topics(db):
    debate_prompt = "Generate 5 thought-provoking debate topics based on the content of the document."
    debate_topics, _ = ask_question(debate_prompt, db)
    print(debate_topics)

def collaborative_document_editing(db):
    print("Welcome to the collaborative document editing session!")
    current_text, _ = ask_question("Summarize the main points of the document in a few paragraphs.", db)
    print(current_text)
    
    while True:
        edit = input("Suggest an edit or addition (or type 'exit' to end): ")
        if edit.lower() == 'exit':
            break
        updated_text, _ = ask_question(f"Update the following text based on this edit: {edit}\n\nCurrent text:\n{current_text}", db)
        current_text = updated_text
        print("\nUpdated text:")
        print(current_text)

def generate_infographic_content(db):
    infographic_prompt = "Create content for an infographic that visually represents the key information from the document. Include main points, statistics, and suggested visual elements."
    infographic_content, _ = ask_question(infographic_prompt, db)
    print(infographic_content)

def main():
    print("Welcome to the advanced document analysis and interaction tool!")
    source_type = input("Enter source type (wikipedia/pdf/hacker_news/directory/youtube/image/audio/video): ")
    source_path = input("Enter source path: ")
    
    chunks = load_and_process_data(source_type, source_path)
    db = create_vector_store(chunks)
    
    while True:
        print("\nAvailable actions:")
        print("1. Analyze token distribution")
        print("2. Generate document summary")
        print("3. Extract key concepts")
        print("4. Interactive Q&A session")
        print("5. Analyze document structure")
        print("6. Perform topic modeling")
        print("7. Extract named entities")
        print("8. Generate and take a quiz")
        print("9. Generate word cloud")
        print("10. Extract keywords")
        print("11. Generate mind map")
        print("12. Generate timeline")
        print("13. Generate analogies")
        print("14. Semantic search")
        print("15. Generate visual summary")
        print("16. Translate document")
        print("17. Generate study guide")
        print("18. Generate counter-arguments")
        print("19. Generate interdisciplinary connections")
        print("20. Generate future scenarios")
        print("21. Generate ethical analysis")
        print("22. Interactive storytelling")
        print("23. Generate multimedia presentation outline")
        print("24. Generate debate topics")
        print("25. Collaborative document editing")
        print("26. Generate infographic content")
        print("27. Exit")
        
        choice = input("Enter your choice (1-27): ")
        
        if choice == '1':
            analyze_token_distribution(chunks)
        elif choice == '2':
            summarize_document(db)
        elif choice == '3':
            extract_key_concepts(db)
        elif choice == '4':
            interactive_qa_session(db)
        elif choice == '5':
            analyze_document_structure(chunks)
        elif choice == '6':
            topic_modeling(chunks)
        elif choice == '7':
            extract_entities(chunks)
        elif choice == '8':
            generate_quiz(db)
        elif choice == '9':
            generate_word_cloud(chunks)
        elif choice == '10':
            extract_keywords(chunks)
        elif choice == '11':
            generate_mind_map(db)
        elif choice == '12':
            generate_timeline(db)
        elif choice == '13':
            generate_analogies(db)
        elif choice == '14':
            query = input("Enter your search query: ")
            semantic_search(query, db)
        elif choice == '15':
            generate_visual_summary(db)
        elif choice == '16':
            target_lang = input("Enter target language code (e.g., 'es' for Spanish): ")
            translated_chunks = translate_document(chunks, target_lang)
            print("Document translated. First chunk:")
            print(translated_chunks[0].page_content)
        elif choice == '17':
            generate_study_guide(db)
        elif choice == '18':
            generate_counter_arguments(db)
        elif choice == '19':
            generate_interdisciplinary_connections(db)
        elif choice == '20':
            generate_future_scenarios(db)
        elif choice == '21':
            generate_ethical_analysis(db)
        elif choice == '22':
            interactive_storytelling(db)
        elif choice == '23':
            generate_multimedia_presentation(db)
        elif choice == '24':
            generate_debate_topics(db)
        elif choice == '25':
            collaborative_document_editing(db)
        elif choice == '26':
            generate_infographic_content(db)
        elif choice == '27':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()   
