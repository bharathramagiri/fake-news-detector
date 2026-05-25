import streamlit as st
import pickle
import time

# --- 1. SET UP THE PAGE & THEME ---
st.set_page_config(
    page_title="TruthLens | Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# --- 2. SIDEBAR INTERFACE ---
st.sidebar.title("About the App")
st.sidebar.info(
    "This is a Machine Learning model trained to detect fake news. "
    "It uses a **Passive-Aggressive Classifier** and **TF-IDF Vectorization** to analyze text patterns."
)
st.sidebar.warning("Note: AI models are not 100% accurate. Always verify breaking news with multiple reliable sources.")

# --- 3. MAIN PAGE HEADER ---
# Using some basic HTML/CSS to center the title and make it pop
st.markdown("<h1 style='text-align: center;'>📰 TruthLens</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>AI-Powered Fake News Detector</h4>", unsafe_allow_html=True)
st.divider() # Adds a neat horizontal line

# --- 4. LOAD THE MODEL (WITH CACHING) ---
# st.cache_resource keeps the model in memory so it doesn't reload every time you click a button
@st.cache_resource 
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error("Model files not found! Please upload model.pkl and vectorizer.pkl.")
    st.stop()

# --- 5. USER INPUT AREA ---
st.write("### Paste the article text below:")
user_input = st.text_area(
    label="Article Text",
    label_visibility="collapsed", # Hides the small default label for a cleaner look
    height=250, 
    placeholder="e.g., 'BREAKING: Scientists discover a new planet made entirely of diamond...'"
)

# Use columns to make the button look smaller and centered
col1, col2, col3 = st.columns([1, 1, 1]) 
with col2:
    analyze_button = st.button("🔍 Analyze Article", use_container_width=True)

# --- 6. PREDICTION & RESULTS ---
if analyze_button:
    if user_input.strip() == "":
        st.warning("Please paste an article first to analyze!")
    else:
        # Show a loading spinner so the user knows the AI is thinking
        with st.spinner("Analyzing text patterns..."):
            time.sleep(1) # Adds a 1-second delay so the spinner is actually visible
            
            # Predict
            vectorized_input = vectorizer.transform([user_input])
            prediction = model.predict(vectorized_input)[0]
            
            st.divider()
            st.write("### Analysis Result:")
            
            # Display custom result cards
            if prediction == True:
                st.success("✅ **Likely REAL News**")
                st.write("> *This text matches the linguistic patterns of credible journalism found in our training dataset.*")
            else:
                st.error("🚨 **Likely FAKE News**")
                st.write("> *This text exhibits sensationalism or specific word patterns commonly found in fabricated articles.*")
