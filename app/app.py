import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import time
import pandas as pd 
from web_scraping.scraping import scrape_website
import speech_recognition as sr

if "scraped_text" not in st.session_state:
    st.session_state.scraped_text = ""

# ==============================
# Load ML components
# ===============================
intent_model = joblib.load("../Notebooks/intent_model.pkl")
intent_tfidf = joblib.load("../Notebooks/tfidf.pkl")
intent_le = joblib.load("../Notebooks/label_encoder.pkl")

sentiment_model = joblib.load("../Notebooks/sentiment_model.pkl")
sentiment_tfidf = joblib.load("../Notebooks/sentiment_tfidf.pkl")
sentiment_le = joblib.load("../Notebooks/sentiment_label_encoder.pkl")

# Sentiment mapping
sentiment_map = {
    -1: "Negative 😞",
    0: "Neutral 😐",
    1: "Positive 🙂"
}

def predict_sentiment(text):
    """
    Predict sentiment for given text using the existing sentiment model.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        tuple: (sentiment_label, confidence_score)
    """
    # Transform text using TF-IDF
    sentiment_vec = sentiment_tfidf.transform([text])
    
    # Predict sentiment
    sentiment_pred = sentiment_model.predict(sentiment_vec)
    
    # Get confidence score
    probs = sentiment_model.predict_proba(sentiment_vec)[0]
    confidence = max(probs)
    
    # Map prediction to label
    sentiment_label = sentiment_map.get(int(sentiment_pred[0]), "Unknown")
    
    # Apply negative keyword override
    if any(word in text.lower() for word in NEGATIVE_KEYWORDS):
        sentiment_label = "Negative 😞"
    
    return sentiment_label, confidence

def recognize_speech():
    """
    Improved speech recognition function with optimized parameters for complete sentence capture.
    
    Returns:
        str: Recognized text from speech
        
    Raises:
        sr.UnknownValueError: When speech is not recognized
        sr.RequestError: When there's an API error
        sr.WaitTimeoutError: When no speech is detected within timeout
        OSError: When microphone is not available
    """
    # Initialize speech recognizer
    r = sr.Recognizer()
    
    # Adjust pause threshold to 1.5 seconds to allow longer pauses before stopping
    # This prevents cutting off the last words when speaker pauses naturally
    r.pause_threshold = 1.5
    
    # Use microphone as audio source
    with sr.Microphone() as source:
        # Adjust for ambient noise to improve recognition accuracy
        r.adjust_for_ambient_noise(source, duration=1)
        
        # Listen for audio input with optimized parameters:
        # - timeout=5: Wait up to 5 seconds for speech to start
        # - phrase_time_limit=8: Allow up to 8 seconds of continuous speech
        # This ensures complete sentences are captured without premature cutoff
        audio = r.listen(source, timeout=5, phrase_time_limit=8)
    
    # Convert speech to text using Google Web Speech API
    text = r.recognize_google(audio)
    
    # Debug: Print recognized text for troubleshooting
    print(f"DEBUG: Recognized text: '{text}'")
    
    return text

# ===============================
NEGATIVE_KEYWORDS = [
    "sucks", "hate", "terrible", "awful", "worst",
    "bad", "boring", "annoying", "irritating",
    "useless", "pathetic", "disgusting"
]

# ===============================
# Intent-based responses
# ===============================
INTENT_RESPONSES = {
    "PlayMusic": "🎵 Sure! Playing something you might like.",
    "GetWeather": "🌤️ Let me check the weather for you.",
    "BookRestaurant": "🍽️ I can help you book a restaurant.",
    "AddToPlaylist": "➕ Added to your playlist!",
    "SearchCreativeWork": "🔍 Searching for creative content.",
    "SearchScreeningEvent": "🎬 Looking up screening events."
}

# ===============================
# Page configuration
# ===============================
st.set_page_config(
    page_title="Intelligent NLP BoT",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>

/* -------- Main App Background -------- */
.stApp {
      background-color: #0e1117;
    color: #e5e7eb;
}

/* -------- Chat Message Containers -------- */
div[data-testid="stChatMessage"] {
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 10px;
    max-width: 90%;
}

/* -------- User Messages -------- */
div[data-testid="stChatMessage"][aria-label="user"] {
    background-color: #1f6feb;
    color: white;
    margin-left: auto;
}

/* -------- Bot Messages -------- */
div[data-testid="stChatMessage"][aria-label="assistant"] {
    background-color: #2d2d2d;
    color: #eaeaea;
    margin-right: auto;
}

/* -------- Buttons -------- */
button[kind="primary"], button[kind="secondary"] {
    border-radius: 10px !important;
    padding: 0.6em 1em !important;
    font-weight: 600 !important;
}

/* -------- Chat Input Box -------- */
textarea {
    border-radius: 12px !important;
    background-color: #1e1e1e !important;
    color: white !important;
}

/* -------- Sidebar -------- */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #374151;
}

/* -------- Headings -------- */
h1, h2, h3 {
    color: #e5e7eb;
}

/* -------- Divider -------- */
hr {
    border: none;
    height: 1px;
    background-color: #374151;
}

</style>
""", unsafe_allow_html=True)



st.title("🤖 Intelligent NLP BoT")
st.caption("Intent & Sentiment Classification using Machine Learning")

st.markdown("""
<div style="
    margin-top: 10px;
    margin-bottom: 20px;
    padding: 14px 18px;
    border-radius: 12px;
    background-color: #eef2ff;
    color: #1e3a8a;
    font-size: 15px;
    font-weight: 500;
">
🤖 <b>Intelligent NLP Bot</b><br>
Understand user <b>intent</b> and <b>sentiment</b> using Machine Learning.
</div>
""", unsafe_allow_html=True)



# ===============================
# Clickable suggestion buttons
# ===============================
# -------------------------------
# Clickable suggestion buttons
# -------------------------------

st.markdown("### 💡 Try one of these:")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎵 Play some music", use_container_width=True):
        st.session_state.pending_input = "Play some music"
        st.rerun()

with col2:
    if st.button("🌤️ What's the weather today?", use_container_width=True):
        st.session_state.pending_input = "What's the weather today?"
        st.rerun()

with col3:
    if st.button("😞 This song sucks", use_container_width=True):
        st.session_state.pending_input = "This song sucks, play something else"
        st.rerun()


# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("⚙️ Controls")

    app_mode = st.selectbox(
         "Choose Feature",
        ["Intent & Sentiment Analysis", "Web Scraping", "🎙️ Speech Analyzer"]
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    debug_mode = st.checkbox("🔧 Show Debug Info")

    st.markdown("---")
    # st.markdown(
    #     "### 🤖 About this Bot\n"
    #     "- Intent Classification\n"
    #     "- Sentiment Analysis\n"
    #     "- Confidence-based fallback\n"
    #     "- Rule-based NLP override"


    st.sidebar.markdown("""
    <div style="
    margin-top: 15px;
    padding: 14px;
    border-radius: 10px;
    background-color: #f1f5f9;
    color: #111827;
    font-size: 14px;
    ">
    <b>🧠 About This Bot</b><br><br>
    • Intent Classification<br>
    • Sentiment Analysis<br>
    • Confidence-based fallback<br>
    • Rule-based NLP override<br>
    • Scraping Data<br>
    </div>
        """, unsafe_allow_html=True)


if app_mode == "Intent & Sentiment Analysis":


# ===============================
# Initialize chat history
# ===============================

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None

# ===============================
# Welcome message (ONLY ONCE)
# ===============================
    if len(st.session_state.chat_history) == 0:
        st.session_state.chat_history.append((
            "assistant",
            "👋 **Hi! I'm your Intelligent NLP BoT** 🤖\n\n"
            "I can help you with:\n"
            "- 🎵 Playing music\n"
            "- 🌤️ Weather info\n"
            "- 🍽️ Booking restaurants\n"
            "- 😊 Understanding sentiment\n\n"
            "Try typing something like:\n"
            "`Play some music` or `This song sucks`"
        ))

# ===============================
# Display chat history
# ===============================

    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender):
            st.write(message)

# ===============================
# Chat input
# ===============================

    user_input = st.chat_input("Type your message here...")

    if st.session_state.pending_input:
        user_input = st.session_state.pending_input
        st.session_state.pending_input = None

# ===============================
# Prediction logic
# ===============================

    CONFIDENCE_THRESHOLD = 0.45

    if user_input:
        # Store user message
        st.session_state.chat_history.append(("user", user_input))

    # ===============================
    # BOT THINKING INDICATOR
    # ===============================

        with st.spinner("🤖 Typing Response..."):
            time.sleep(0.5)  # ensures spinner is visible

            # -------- INTENT PREDICTION --------
            intent_vec = intent_tfidf.transform([user_input])
            intent_pred = intent_model.predict(intent_vec)
            intent_label = intent_le.inverse_transform(intent_pred)[0]

            intent_probs = intent_model.predict_proba(intent_vec)
            intent_confidence = intent_probs.max()

        # -------- SENTIMENT PREDICTION --------
            sentiment_label, sentiment_confidence = predict_sentiment(user_input)

            # -------- SENTIMENT COLOR --------
            if "Negative" in sentiment_label:
                sentiment_display = f"🔴 **Sentiment:** {sentiment_label}"
            elif "Positive" in sentiment_label:
                sentiment_display = f"🟢 **Sentiment:** {sentiment_label}"
            else:
                sentiment_display = f"🟡 **Sentiment:** {sentiment_label}"

            # -------- CONFIDENCE LANGUAGE --------
            if intent_confidence < 0.6:
                confidence_text = "⚠️ I might not be fully sure."
            else:
                confidence_text = "✅ I’m confident about this."

            # -------- FALLBACK --------
            if intent_confidence < CONFIDENCE_THRESHOLD:
                bot_response = (
                    "🤔 I'm not very confident about your request.\n\n"
                    "Could you please rephrase it?"
                )
            else:
                intent_response = INTENT_RESPONSES.get(
                    intent_label,
                    "🤖 I understood your request."
                )

                bot_response = (
                    f"{intent_response}\n\n"
                    f"🎯 **Intent:** {intent_label}\n"
                    f"{sentiment_display}\n"
                    f"📊 **Confidence:** {intent_confidence:.2f}\n"
                    f"{confidence_text}"
                )

            # -------- DEBUG --------
            if debug_mode:
                bot_response += (
                    f"\n\n🔧 Debug Info:\n"
                    f"- Sentiment confidence: {sentiment_confidence:.2f}\n"
                    f"- Intent probabilities: {intent_probs}"
                )

        # Store bot response
        st.session_state.chat_history.append(("assistant", bot_response))

        # Refresh UI
        st.rerun()



        st.markdown("""
        <hr style="margin-top:40px; margin-bottom:10px;">
        <center style="color:#6b7280; font-size:13px;">
        Built with ❤️ using Python, Machine Learning & Streamlit
        </center>
        """, unsafe_allow_html=True)



#-------------------------------------- New Web Scraping Code --------------------------------------

elif app_mode == "Web Scraping":
    st.title("🌐 Web Scraping Module")
    st.caption("Extract text from public websites, download it, or analyze sentiment")

    # ---- URL Input ----
    url = st.text_input(
        "🌐 Enter Website URL",
        placeholder="https://example.com"
    )

    # ---- Scrape Button ----
    if st.button("Scrape Website"):
        if url.strip() == "":
            st.warning("⚠️ Please enter a valid URL.")
        else:
            with st.spinner("🔍 Scraping website..."):
                result = scrape_website(url)
                st.session_state.scraped_result = result

            # ---- SUCCESS CASE ----
            # st.write("DEBUG TYPE:", type(result))
            # st.write("DEBUG VALUE:", result)

            if isinstance(result, dict) and result.get("success"):
                st.success("✅ Scraping completed!")

                # ---- Preview ----
                st.markdown("### 🔍 Preview")
                st.write(result["text"][:400] + "...")

                # ---- Expand Full Text ----
                with st.expander("📄 View Full Scraped Text"):
                    st.text_area(
                        "Extracted Text",
                        result["text"],
                        height=300
                    )

                # ---- Metrics ----
                st.markdown("### 📊 Scraping Summary")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Characters", len(result["text"]))

                with col2:
                    st.metric("Word Count", result["word_count"])

                # ---- Download Button ----
                st.download_button(
                    label="📥 Download Text",
                    data=result["text"],
                    file_name="scraped_text.txt",
                    mime="text/plain"
                )

            # ---- ERROR CASE ----
            else:
                if isinstance(result, dict):
                    st.error(result.get("error", "Unknown error"))
                else:
                    st.error("⚠️ Unexpected response from scraper")


            # -------------------------------
            # DOWNLOAD OPTIONS
            # -------------------------------
            st.markdown("### 📥 Download Scraped Text")

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="⬇️ Download as TXT",
                    data=st.session_state.scraped_text,
                    file_name="scraped_text.txt",
                    mime="text/plain"
                )

            with col2:
                df = pd.DataFrame({"Text": [st.session_state.scraped_text]})
                st.download_button(
                    label="⬇️ Download as CSV",
                    data=df.to_csv(index=False),
                    file_name="scraped_text.csv",
                    mime="text/csv"
                )

                st.markdown("### 🔄 Actions")

    else:
        st.info(
            "ℹ️ No data scraped yet.\n\n"
            "👉 Enter a website URL above and click **Scrape Website** to begin."
        )
  

    if st.button("Clear Scraped Data"):
        st.session_state.scraped_text = ""
        st.success("Scraped data cleared.")
        st.rerun()

elif app_mode == "🎙️ Speech Analyzer":
    st.title("🎙️ Speech Analyzer")
    st.caption("Convert speech to text and analyze sentiment")
    
    # Installation instructions:
    # pip install SpeechRecognition pyaudio
    
    st.markdown("""
    ### 🎤 How it works:
    1. Click "Start Recording" to begin voice input
    2. Speak clearly into your microphone
    3. The system will transcribe your speech and analyze its sentiment
    """)
    
    # Start Recording Button
    if st.button("🎤 Start Recording", type="primary"):
        with st.spinner("🎤 Listening... Please speak now."):
            try:
                # Call the improved speech recognition function
                text = recognize_speech()
                
                # Display success and transcribed text
                st.success("✅ Speech recognized successfully!")
                st.markdown("### 📝 Transcribed Text")
                st.text_area("Your speech:", text, height=100, disabled=True)
                
                # Analyze sentiment
                st.markdown("### 😊 Sentiment Analysis")
                sentiment, confidence = predict_sentiment(text)
                
                # Display sentiment with appropriate color
                if "Negative" in sentiment:
                    st.error(f"**Sentiment:** {sentiment}")
                elif "Positive" in sentiment:
                    st.success(f"**Sentiment:** {sentiment}")
                else:
                    st.warning(f"**Sentiment:** {sentiment}")
                
                # Display confidence score if available
                if confidence is not None:
                    st.info(f"**Confidence Score:** {confidence:.2f}")
                
            except sr.RequestError as e:
                st.error(f"❌ API Error: Could not connect to speech recognition service. {e}")
                st.info("💡 **Tip:** Check your internet connection and try again.")
            except sr.UnknownValueError:
                st.warning("⚠️ Speech not recognized. Please speak clearly and try again.")
                st.info("💡 **Tip:** Speak louder, reduce background noise, or try the retry button below.")
            except sr.WaitTimeoutError:
                st.warning("⚠️ No speech detected within timeout period. Please try again.")
                st.info("💡 **Tip:** Make sure your microphone is working and start speaking promptly.")
            except OSError as e:
                st.error(f"❌ Microphone Error: {e}. Please check your microphone setup.")
                st.info("💡 **Tip:** Ensure microphone permissions are granted and device is connected.")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {e}")
                st.info("💡 **Tip:** Try restarting the app or check the console for more details.")
    
    # Retry Button
    st.markdown("---")
    if st.button("🔄 Retry Recording"):
        st.rerun()
    
    # Additional information
    st.markdown("""
    ### 📋 Notes:
    - Make sure your microphone is enabled and working
    - Speak clearly and at a normal pace
    - The system uses Google's Web Speech API (requires internet connection)
    - Processing may take a few seconds
    - **Improved accuracy:** The system now captures complete sentences without cutting off last words
    """)






