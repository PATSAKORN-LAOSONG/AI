import streamlit as st

st.set_page_config(page_title="AI Tutor ม.6", layout="centered")

st.title("🤖 AI แนะนำบทเรียนสำหรับ ม.6")

# ----------------------------
# เก็บสถานะ
# ----------------------------
if "score" not in st.session_state:
    st.session_state.score = 0

if "question_number" not in st.session_state:
    st.session_state.question_number = 0

if "difficulty" not in st.session_state:
    st.session_state.difficulty = 1

# ----------------------------
# คลังคำถาม
# ----------------------------
questions = {
    1: [
        {"q": "2x + 3 = 7 ค่า x เท่ากับอะไร?", "a": "2"},
        {"q": "5 + 7 = ?", "a": "12"},
    ],
    2: [
        {"q": "แก้สมการ 3x - 5 = 10", "a": "5"},
        {"q": "พื้นที่วงกลมสูตรคืออะไร?", "a": "pi r^2"},
    ],
    3: [
        {"q": "ลิมิตของ x→0 (sinx/x) เท่ากับอะไร?", "a": "1"},
        {"q": "อนุพันธ์ของ x^2 คืออะไร?", "a": "2x"},
    ],
}

# ----------------------------
# ฟังก์ชันถามคำถาม
# ----------------------------
def ask_question():
    level = st.session_state.difficulty
    q_list = questions[level]
    index = st.session_state.question_number % len(q_list)
    return q_list[index]

# ----------------------------
# เริ่มระบบ
# ----------------------------
if st.button("เริ่มทำแบบทดสอบ"):
    st.session_state.score = 0
    st.session_state.question_number = 0
    st.session_state.difficulty = 1

if st.session_state.question_number < 5:

    question = ask_question()
    st.write(f"📘 คำถามระดับ {st.session_state.difficulty}")
    st.write(question["q"])

    user_answer = st.text_input("พิมพ์คำตอบของคุณ:")

    if st.button("ส่งคำตอบ"):
        if user_answer.lower() == question["a"].lower():
            st.success("✅ ถูกต้อง!")
            st.session_state.score += 1
            st.session_state.difficulty = min(3, st.session_state.difficulty + 1)
        else:
            st.error("❌ ยังไม่ถูก")
            st.session_state.difficulty = max(1, st.session_state.difficulty - 1)

        st.session_state.question_number += 1
        st.rerun()

else:
    st.write("🎯 ผลลัพธ์ของคุณ")
    final_score = st.session_state.score
    percentage = (final_score / 5) * 100
    st.write(f"คะแนนรวม: {percentage}%")

    if percentage < 50:
        st.warning("คุณควรทบทวนพื้นฐานคณิตศาสตร์เพิ่มเติม")
    elif percentage < 80:
        st.info("คุณอยู่ระดับปานกลาง ควรฝึกทำโจทย์เพิ่ม")
    else:
        st.success("ยอดเยี่ยม! คุณพร้อมสำหรับโจทย์ระดับสูง")
