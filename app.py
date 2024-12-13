import streamlit as st
import re
import nltk
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Check if pickle file exists, else create it
PICKLE_FILE = "keywords.pkl"
if not os.path.exists(PICKLE_FILE):
    # Define default keywords
    data = {
        "SKILLS_KEYWORDS": ["python", "javascript", "react", "node.js", "machine learning", "data analysis"],
        "EXPERIENCE_KEYWORDS": ["developed", "built", "designed", "managed", "implemented", "led", "collaborated"],
        "ACHIEVEMENTS_KEYWORDS": ["award", "certification", "ranked", "published", "innovated", "honor"],
        "PROJECTS_KEYWORDS": ["project", "system", "platform", "application", "tool", "framework"]
    }
    with open(PICKLE_FILE, "wb") as file:
        pickle.dump(data, file)
    print(f"Created default pickle file: {PICKLE_FILE}")

# Load keywords from pickle file
with open(PICKLE_FILE, "rb") as file:
    keywords = pickle.load(file)

# Define scoring function
def calculate_score(text, keywords):
    """Calculate score based on the presence of keywords."""
    if not text.strip():
        return 0  # Return 0 if the text is empty
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words and word.isalnum()]
    vectorizer = CountVectorizer().fit_transform([" ".join(filtered_words), " ".join(keywords)])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    return round(similarity[0][1] * 100, 2)  # Score as a percentage

# Extract sections from resume
def extract_section(text, section_name):
    """Extract text corresponding to a section."""
    pattern = rf"(?i){section_name}\s*.*?(?=(\n[A-Z]|$))"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0).replace(section_name, "").strip()
    return ""

# Streamlit app
st.title("AI Resume Scorer")
st.write("Upload a resume PDF to extract and score each section.")

uploaded_file = st.file_uploader("Upload Resume (PDF File)", type=["pdf"])

if uploaded_file:
    # Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()

    # Extract and score sections
    skills_text = extract_section(resume_text, "SKILLS")
    experience_text = extract_section(resume_text, "EXPERIENCE")
    achievements_text = extract_section(resume_text, "ACHIEVEMENTS")
    projects_text = extract_section(resume_text, "PROJECTS")

    # Calculate scores
    skills_score = calculate_score(skills_text, keywords["SKILLS_KEYWORDS"])
    experience_score = calculate_score(experience_text, keywords["EXPERIENCE_KEYWORDS"])
    achievements_score = calculate_score(achievements_text, keywords["ACHIEVEMENTS_KEYWORDS"])
    projects_score = calculate_score(projects_text, keywords["PROJECTS_KEYWORDS"])

    total_score = round((skills_score + experience_score + achievements_score + projects_score) / 4, 2)

    # Display scores
    st.subheader("Section Scores")
    st.write(f"**Skills Score:** {skills_score}%")
    st.write(f"**Experience Score:** {experience_score}%")
    st.write(f"**Achievements Score:** {achievements_score}%")
    st.write(f"**Projects Score:** {projects_score}%")

    st.subheader("Total Score")
    st.write(f"**Total Score:** {total_score}%")
