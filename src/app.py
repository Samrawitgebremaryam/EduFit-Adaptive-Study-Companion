import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Load models
try:
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    kmeans = joblib.load(os.path.join(MODELS_DIR, 'kmeans.joblib'))
    rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
    st.write("Models loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error: Model file not found. Please ensure models are saved in 'models' directory. ({str(e)})")
    st.stop()

# Sidebar
st.sidebar.title("EduFit: Adaptive Study Companion")
page = st.sidebar.selectbox("Choose an Option", ["Get Learning Style", "Get Study Strategies", "Get Resources"])

# Learning style questions (VARK-based)
learning_style_questions = [
    {
        "question": "When learning something new, I prefer to:",
        "options": {
            "See diagrams, charts, or videos": "Visual",
            "Hear explanations or discuss them": "Auditory",
            "Try it hands-on or move around": "Kinesthetic",
            "Read texts or write notes": "Reading"
        }
    },
    {
        "question": "To remember information, I usually:",
        "options": {
            "Visualize it in my mind": "Visual",
            "Repeat it aloud or talk it through": "Auditory",
            "Practice it or touch objects": "Kinesthetic",
            "Write it down or read it repeatedly": "Reading"
        }
    },
    {
        "question": "I understand best when I can:",
        "options": {
            "Watch a demonstration or see pictures": "Visual",
            "Listen to a lecture or podcast": "Auditory",
            "Participate in activities or experiments": "Kinesthetic",
            "Take detailed notes or read instructions": "Reading"
        }
    },
    {
        "question": "When studying, I prefer:",
        "options": {
            "Color-coded notes or mind maps": "Visual",
            "Study groups or audio recordings": "Auditory",
            "Role-playing or building models": "Kinesthetic",
            "Textbooks or writing summaries": "Reading"
        }
    },
    {
        "question": "I find it easiest to learn by:",
        "options": {
            "Looking at infographics or slides": "Visual",
            "Explaining it to someone else": "Auditory",
            "Moving or using physical tools": "Kinesthetic",
            "Reading articles or lists": "Reading"
        }
    }
]

# Learning style descriptions and advice
learning_style_info = {
    'Visual': {
        'description': 'You learn best through images, diagrams, and spatial organization.',
        'advice': 'Use mind maps, color-coded notes, and videos. Watch Khan Academy or draw concepts. Study in visually stimulating environments.'
    },
    'Auditory': {
        'description': 'You learn best through sound, discussion, and verbal explanations.',
        'advice': 'Join study groups, listen to podcasts, or record yourself explaining concepts. Use apps like Audible or explain ideas aloud.'
    },
    'Kinesthetic': {
        'description': 'You learn best through touch, movement, and hands-on activities.',
        'advice': 'Use physical models, role-play, or study while moving (e.g., walking). Try interactive tools like PhET simulations or build study aids.'
    },
    'Reading': {
        'description': 'You learn best through reading texts and writing notes.',
        'advice': 'Summarize readings, use Cornell Notes, or rewrite concepts. Read textbooks or articles on platforms like Nature or Medium.'
    }
}

# Resource fetching function
def fetch_resources(learning_style, topic):
    topic = topic.lower()
    resources = {
        'Visual': {
            'calculus': "Khan Academy: Calculus (https://www.khanacademy.org/math/calculus-1)",
            'physics': "CrashCourse: Physics (https://www.youtube.com/playlist?list=PL8dPuuaLjXtN0ge7yDk_UA0ldZJdhwkoV)",
            'machine learning': "freeCodeCamp: Machine Learning (https://www.youtube.com/watch?v=Eo9L-TWFZ3I)",
            'biology': "Bozeman Science: Biology (https://www.youtube.com/user/bozemanscience)",
            'history': "CrashCourse: World History (https://www.youtube.com/playlist?list=PLBDA2E52FB1EF80C9)",
            'default': "Khan Academy: General (https://www.khanacademy.org)"
        },
        'Auditory': {
            'calculus': "The Math Factor: Math Topics (http://mathfactor.uark.edu)",
            'physics': "StarTalk Radio: Physics (https://www.startalkradio.net)",
            'machine learning': "Data Skeptic: Machine Learning (https://dataskeptic.com)",
            'biology': "This Week in Science: Biology (https://www.twis.org)",
            'history': "History Extra: World History (https://www.historyextra.com/podcasts)",
            'default': "BBC Learning: General (https://www.bbc.co.uk/programmes/p02nrsln)"
        },
        'Kinesthetic': {
            'calculus': "GeoGebra: Calculus (https://www.geogebra.org/calculus)",
            'physics': "PhET: Physics Simulations (https://phet.colorado.edu/en/simulations/category/physics)",
            'machine learning': "TensorFlow Playground: ML (https://playground.tensorflow.org)",
            'biology': "PhET: Biology Simulations (https://phet.colorado.edu/en/simulations/category/biology)",
            'history': "Smithsonian Learning Lab: History (https://learninglab.si.edu)",
            'default': "PhET: General (https://phet.colorado.edu)"
        },
        'Reading': {
            'calculus': "BetterExplained: Calculus (https://betterexplained.com/articles/a-gentle-introduction-to-calculus/)",
            'physics': "Scientific American: Physics (https://www.scientificamerican.com/physics/)",
            'machine learning': "Medium: ML Basics (https://medium.com/topic/machine-learning)",
            'biology': "Nature: Biology (https://www.nature.com/subjects/biology)",
            'history': "History.com: Topics (https://www.history.com/topics)",
            'default': "SparkNotes: General (https://www.sparknotes.com)"
        }
    }
    return resources.get(learning_style, {}).get(topic, resources[learning_style]['default'])

# Page: Get Learning Style
if page == "Get Learning Style":
    st.header("Discover Your Learning Style")
    st.write("Answer the following questions to find out how you learn best.")

    # Collect answers
    answers = {}
    for i, q in enumerate(learning_style_questions):
        answers[q['question']] = st.radio(q['question'], list(q['options'].keys()), key=f"q{i}")

    if st.button("Submit"):
        # Score responses
        scores = {'Visual': 0, 'Auditory': 0, 'Kinesthetic': 0, 'Reading': 0}
        for question, answer in answers.items():
            for q in learning_style_questions:
                if q['question'] == question:
                    style = q['options'][answer]
                    scores[style] += 1
                    break

        # Determine learning style
        learning_style = max(scores, key=scores.get)
        st.subheader("Your Learning Style")
        st.write(f"**{learning_style}**")
        st.write(f"**Description**: {learning_style_info[learning_style]['description']}")
        st.write(f"**Advice**: {learning_style_info[learning_style]['advice']}")

# Page: Get Study Strategies
elif page == "Get Study Strategies":
    st.header("Get Personalized Study Strategies")
    st.write("Provide your study habits to receive tailored strategy recommendations.")

    # User input
    studytime = st.slider("Weekly Study Time (hours)", 1, 40, 10)
    absences = st.slider("Absences per semester", 0, 20, 0)
    pref_resource = st.selectbox("Preferred Resource", 
                                 options=[1, 2, 3, 4],
                                 format_func=lambda x: {1: "Online Videos", 2: "Textbooks", 3: "Interactive Tools", 4: "Podcasts"}[x])
    daily_study_hours = st.slider("Daily Study Hours", 0.5, 10.0, 2.0)
    subject = st.selectbox("Subject", 
                           options=[1, 2, 3],
                           format_func=lambda x: {1: "Math", 2: "Science", 3: "Other"}[x])

    # Predict learning style
    if st.button("Generate Study Strategy"):
        input_data = pd.DataFrame({
            'studytime': [studytime],
            'absences': [absences],
            'pref_resource': [pref_resource],
            'daily_study_hours': [daily_study_hours],
            'subject': [subject]
        })

        try:
            X_scaled = scaler.transform(input_data[['studytime', 'absences', 'pref_resource', 'daily_study_hours']])
            learning_style = kmeans.predict(X_scaled)[0]
            learning_style_map = {0: 'Visual', 1: 'Auditory', 2: 'Kinesthetic', 3: 'Reading'}
            learning_style_name = learning_style_map[learning_style]
        except Exception as e:
            st.error(f"Error predicting learning style: {str(e)}")
            st.stop()

        # Add learning style to input
        input_data['learning_style'] = learning_style
        features_for_rf = ['studytime', 'absences', 'pref_resource', 'daily_study_hours', 'subject', 'learning_style']
        try:
            strategy = rf.predict(input_data[features_for_rf])[0]
            strategy_map = {0: 'Pomodoro', 1: 'Active Recall', 2: 'Spaced Repetition', 3: 'Note-Taking'}
            strategy_name = strategy_map[strategy]
        except Exception as e:
            st.error(f"Error predicting strategy: {str(e)}")
            st.stop()

        # Display results
        st.subheader("Your Personalized Study Strategy")
        st.write(f"**Learning Style**: {learning_style_name}")
        st.write(f"**Recommended Strategy**: {strategy_name}")
        st.write(f"**Advice**: {learning_style_info[learning_style_name]['advice']}")

# Page: Get Resources
elif page == "Get Resources":
    st.header("Curated Study Resources")
    st.write("Provide your learning style and subject to get tailored resources.")

    # User input
    learning_style = st.selectbox("Learning Style", ["Visual", "Auditory", "Kinesthetic", "Reading"])
    topic = st.text_input("Study Topic (e.g., calculus, physics)", "calculus")

    if st.button("Get Resources"):
        resource = fetch_resources(learning_style, topic)
        st.subheader("Your Curated Resource")
        st.write(f"**Learning Style**: {learning_style}")
        st.write(f"**Topic**: {topic}")
        st.write(f"**Resource**: {resource}")

# Footer
st.write("---")
st.write("Built with EduFit: Adaptive Study Companion")