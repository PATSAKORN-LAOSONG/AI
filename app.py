import streamlit as st
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Competitive Adaptive System", layout="centered")
st.title("🏆 AI Competitive Adaptive Learning System")

# =============================
# SESSION INIT
# =============================
defaults = {
    "level": "medium",
    "score": 0,
    "question_count": 0,
    "correct_streak": 0,
    "wrong_streak": 0,
    "max_correct": 0,
    "max_wrong": 0,
    "level_history": [],
    "finished": False,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

levels = ["easy", "medium", "hard"]

# =============================
# QUESTION BANK
# =============================
question_bank = {
    "easy": [
        {"q": "2+3=?", "choices": ["4","5","6","7"], "answer": "5", "difficulty":1},
        {"q": "4+4=?", "choices": ["6","7","8","9"], "answer": "8", "difficulty":1}
    ],
    "medium": [
        {"q": "อนุพันธ์ x^2?", "choices": ["x","2x","x^2","2"], "answer": "2x", "difficulty":2}
    ],
    "hard": [
        {"q": "อินทิกรัล x^2?", "choices": ["x^3/3","x^2","2x","3x"], "answer": "x^3/3", "difficulty":3}
    ]
}

# =============================
# ML MODEL (Classification)
# =============================
def train_classifier():
    data = {
        "score":[0,1,2,3,4,5],
        "max_correct":[0,1,2,2,3,4],
        "max_wrong":[3,2,2,1,1,0],
        "level":["easy","easy","medium","medium","hard","hard"]
    }
    df = pd.DataFrame(data)
    X = df[["score","max_correct","max_wrong"]]
    y = df["level"]
    model = RandomForestClassifier()
    model.fit(X,y)
    return model

clf = train_classifier()

# =============================
# QUESTION FUNCTION
# =============================
def get_question():
    return random.choice(question_bank[st.session_state.level])

# =============================
# QUIZ LOGIC
# =============================
if not st.session_state.finished:

    q = get_question()
    st.write(f"ข้อที่ {st.session_state.question_count+1} (ระดับ {st.session_state.level})")
    st.write(q["q"])

    ans = st.radio("เลือกคำตอบ", q["choices"])

    if st.button("ส่งคำตอบ"):

        st.session_state.level_history.append(st.session_state.level)

        if ans == q["answer"]:
            st.session_state.score += 1
            st.session_state.correct_streak += 1
            st.session_state.wrong_streak = 0
        else:
            st.session_state.wrong_streak += 1
            st.session_state.correct_streak = 0

        st.session_state.max_correct = max(
            st.session_state.max_correct,
            st.session_state.correct_streak
        )

        st.session_state.max_wrong = max(
            st.session_state.max_wrong,
            st.session_state.wrong_streak
        )

        # Adaptive Rule
        if st.session_state.correct_streak == 2:
            if st.session_state.level != "hard":
                st.session_state.level = levels[levels.index(st.session_state.level)+1]
            st.session_state.correct_streak = 0

        if st.session_state.wrong_streak == 2:
            if st.session_state.level != "easy":
                st.session_state.level = levels[levels.index(st.session_state.level)-1]
            st.session_state.wrong_streak = 0

        st.session_state.question_count += 1

        if st.session_state.question_count >= 5:
            st.session_state.finished = True

        st.rerun()

# =============================
# RESULT SECTION
# =============================
if st.session_state.finished:

    st.success(f"คะแนนรวม: {st.session_state.score}/5")

    features = [[
        st.session_state.score,
        st.session_state.max_correct,
        st.session_state.max_wrong
    ]]

    predicted = clf.predict(features)[0]
    st.info(f"🤖 Classification Result: {predicted}")

    # =============================
    # CLUSTERING
    # =============================
    student_data = np.array([
        [st.session_state.score,
         st.session_state.max_correct,
         st.session_state.max_wrong]
    ])

    kmeans = KMeans(n_clusters=3, random_state=42)
    dummy_data = np.array([
        [0,0,3],
        [2,1,2],
        [5,3,0]
    ])
    kmeans.fit(dummy_data)

    cluster = kmeans.predict(student_data)[0]
    st.write(f"📊 Clustering Group: {cluster}")

    # =============================
    # GRAPH LEVEL CHANGE
    # =============================
    st.write("📈 ระดับที่เปลี่ยนระหว่างทำข้อสอบ")

    level_map = {"easy":1,"medium":2,"hard":3}
    numeric_levels = [level_map[l] for l in st.session_state.level_history]

    plt.plot(numeric_levels)
    plt.ylim(1,3)
    plt.ylabel("Level")
    plt.xlabel("Question")
    st.pyplot(plt)

    if st.button("เริ่มใหม่"):
        for k in defaults.keys():
            st.session_state[k] = defaults[k]
        st.rerun()
