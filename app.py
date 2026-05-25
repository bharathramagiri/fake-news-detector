import streamlit as st
import pickle

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️‍♂️")

st.title("Fake News Detector 🕵️‍♂️")
st.write("Paste the text of a news article below to find out if it is REAL or FAKE.")

# --- 2. LOAD THE SAVED MODEL ---
# We use try/except just in case the files are missing
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model files not found. Please run news.py first to generate model.pkl and vectorizer.pkl.")
    st.stop()

# --- 3. CREATE THE USER INTERFACE ---
# Create a text box for the user to paste the article
user_input = st.text_area("Article Text:", height=250, placeholder="Paste the news story here...")

# Create a submit button
if st.button("Check Authenticity"):
    if user_input.strip() == "":
        st.warning("Please paste an article first!")
    else:
        # --- 4. MAKE THE PREDICTION ---
        # Convert the user's text into numbers using our saved vectorizer
        vectorized_input = vectorizer.transform([user_input])
        
        # Make the prediction using our saved model
        prediction = model.predict(vectorized_input)[0]
        
        # Display the result (Remember: True = Real, False = Fake based on your dataset)
        st.markdown("### Result:")
        if prediction == True:
            st.success("✅ This news article appears to be **REAL**.")
        else:
            st.error("🚨 This news article appears to be **FAKE**.")