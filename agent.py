# import libraries



# -----------------------------------------
# TRAINING DATA (LIST OF SENTENCES)
# -----------------------------------------
training_sentences = [
    # calculator
    "open calculator", "i want to calculate", "start calculator",

    # time
    "what is the time", "tell me the time", "current time please",

    # note
    "make a note", "save this note", "write this down",

    # joke
    "tell me a joke", "make me laugh", "joke please",

    # youtube
    "open youtube", "play youtube", "start youtube",

    # google
    "search on google", "google this", "i want to search something",

    # music
    "play music", "start music", "i want to listen song"
]

# Intent labels matching each sentence
intents = [
    "calculator","calculator","calculator",
    "time","time","time",
    "note","note","note",
    "joke","joke","joke",
    "youtube","youtube","youtube",
    "google","google","google",
    "music","music","music"
]

# -----------------------------------------
# NLP MODEL (TF-IDF VECTOR + SIMILARITY)
# -----------------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)

# -----------------------------------------
# STREAMLIT UI
# -----------------------------------------
st.set_page_config(page_title="AI Intent Agent", layout="centered")
st.title("🤖 Smart AI Agent (Base Template)")
st.write("This is the base version. Students will add functionalities step by step.")

# Input Box
user_input = st.text_input("Your Command:", placeholder="e.g., open calculator, tell me a joke...")

# Detect Intent
if st.button("Detect Intent"):
    if user_input.strip() == "":
        st.warning("Please type something!")
    else:
        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X)
        index = similarity.argmax()
        detected_intent = intents[index]

        st.success(f"Detected Intent → **{detected_intent}**")

        # -----------------------------------------
        # ⚠️ ACTIONS WILL BE ADDED BY STUDENTS HERE
        # -----------------------------------------
       
