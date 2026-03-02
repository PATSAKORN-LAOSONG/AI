import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Learning System M.6", layout="centered")
st.title("🎓 AI ระบบแนะนำบทเรียน ม.6")

# =========================
# โหลดข้อมูลจริง + Train ML
# =========================
@st.cache_resource
def train_model():
    df = pd.read_csv("StudentsPerformance.csv")

    df["total_score"] = (
        df["math score"] +
        df["reading score"] +
        df["writing score"]
    )

    df["level"] = pd.cut(
        df["total_score"],
        bins=[0,150,220,300],
        labels=["ต่ำ","กลาง","สูง"]
    )

    X = df[["math score","reading score","writing score"]]
    y = df["level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy

model, accuracy = train_model()
st.write(f"📊 ความแม่นยำโมเดล: {accuracy:.2f}")

# =========================
# Question Bank (5 ข้อต่อวิชา)
# =========================
question_bank = {
    "Math": [
        {"q":"2+2=?","c":["3","4","5","6"],"a":"4","level":1},
        {"q":"10 ÷ 2 = ?","c":["2","3","4","5"],"a":"5","level":1},
        {"q":"5x=20, x=?","c":["2","3","4","5"],"a":"4","level":2},
        {"q":"พื้นที่สามเหลี่ยมฐาน4สูง5?","c":["10","15","20","25"],"a":"10","level":2},
        {"q":"อนุพันธ์ x^2 ?","c":["2x","x","x^2","1"],"a":"2x","level":3},
    ],
    "Reading": [
        {"q":"Main idea คืออะไร?","c":["ใจความหลัก","คำศัพท์","ไวยากรณ์","ผู้เขียน"],"a":"ใจความหลัก","level":1},
        {"q":"Synonym หมายถึง?","c":["คำตรงข้าม","คำเหมือน","คำกริยา","คำนาม"],"a":"คำเหมือน","level":1},
        {"q":"Inference คือ?","c":["สรุป","อนุมาน","แปลตรงตัว","สะกดคำ"],"a":"อนุมาน","level":2},
        {"q":"Context clue ใช้ทำอะไร?","c":["เดาความหมาย","สรุปเรื่อง","วิเคราะห์","จับเวลา"],"a":"เดาความหมาย","level":2},
        {"q":"Tone ของเรื่องหมายถึง?","c":["เสียง","อารมณ์","ตัวละคร","สถานที่"],"a":"อารมณ์","level":3},
    ],
    "Writing": [
        {"q":"Essay มีกี่ย่อหน้า?","c":["1","2","3","5"],"a":"3","level":1},
        {"q":"Paragraph ต้องมีอะไรหลัก?","c":["Topic sentence","Grammar","Verb","Adverb"],"a":"Topic sentence","level":1},
        {"q":"Topic sentence คือ?","c":["ประโยคหลัก","ประโยครอง","คำเชื่อม","บทสรุป"],"a":"ประโยคหลัก","level":2},
        {"q":"Conclusion ทำหน้าที่?","c":["เปิดเรื่อง","สรุป","ยกตัวอย่าง","ตั้งคำถาม"],"a":"สรุป","level":2},
        {"q":"Thesis statement คือ?","c":["สรุป","แนวคิดหลัก","คำศัพท์","หัวข้อ"],"a":"แนวคิดหลัก","level":3},
    ]
}

# =========================
# Session State
# =========================
if "initialized" not in st.session_state:
    st.session_state.subject = "Math"
    st.session_state.level = 2
    st.session_state.score = 0
    st.session_state.count = 0
    st.session_state.correct_streak = 0
    st.session_state.finished = False
    st.session_state.used_questions = []
    st.session_state.initialized = True

subject = st.selectbox("เลือกวิชา", ["Math","Reading","Writing"])

# รีเซ็ตเมื่อเปลี่ยนวิชา
if subject != st.session_state.subject:
    st.session_state.subject = subject
    st.session_state.level = 2
    st.session_state.score = 0
    st.session_state.count = 0
    st.session_state.correct_streak = 0
    st.session_state.finished = False
    st.session_state.used_questions = []
    if "current_question" in st.session_state:
        del st.session_state.current_question

# =========================
# Adaptive Test
# =========================
if not st.session_state.finished:

    progress = st.session_state.count / 5
    st.progress(progress)

    questions = [
        q for q in question_bank[subject]
        if q["level"] == st.session_state.level
        and q["q"] not in st.session_state.used_questions
    ]

    if questions:
        if "current_question" not in st.session_state:
            st.session_state.current_question = random.choice(questions)

        q = st.session_state.current_question

        st.subheader(f"ข้อที่ {st.session_state.count+1}")
        st.write(f"ระดับความยาก: {st.session_state.level}")
        st.write(q["q"])

        choice = st.radio("เลือกคำตอบ", q["c"], key=f"radio_{st.session_state.count}")

        if st.button("ส่งคำตอบ"):

            if choice == q["a"]:
                st.success("ถูกต้อง!")
                st.session_state.score += 15
                st.session_state.correct_streak += 1

                if st.session_state.correct_streak >= 2:
                    st.session_state.level = min(3, st.session_state.level+1)
                    st.session_state.score += 5
                    st.session_state.correct_streak = 0
            else:
                st.error("ผิด!")
                st.session_state.level = max(1, st.session_state.level-1)
                st.session_state.correct_streak = 0

            st.session_state.used_questions.append(q["q"])
            st.session_state.count += 1

            if st.session_state.count >= 5:
                st.session_state.finished = True

            del st.session_state.current_question
            st.rerun()

# =========================
# Result Section
# =========================
else:
    st.subheader("✅ ทำข้อสอบเสร็จแล้ว")
    st.write("คะแนนที่ได้:", st.session_state.score)

    predicted = model.predict([[st.session_state.score,
                                st.session_state.score,
                                st.session_state.score]])[0]

    st.success(f"🎯 ระดับความเข้าใจ: {predicted}")

    st.subheader("🤖 AI แนะนำบทเรียน")

    if predicted == "ต่ำ":
        st.info("ควรทบทวนพื้นฐาน และทำแบบฝึกหัดระดับง่าย")
    elif predicted == "กลาง":
        st.info("ควรฝึกโจทย์ประยุกต์ และจับเวลา")
    else:
        st.info("พร้อมทำข้อสอบเข้ามหาวิทยาลัย และโจทย์ยาก")

    if st.button("เริ่มใหม่"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
