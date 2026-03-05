# app.py
import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Math Skill AI", layout="centered")

st.title("🧠 ระบบวิเคราะห์จุดอ่อนคณิตศาสตร์ด้วย AI + Radar Chart")

# =========================
# โหลด Dataset และ Train ML (มี fallback ถ้าไฟล์ไม่พบ)
# =========================
@st.cache_resource
def train_model():
    csv_path = "math_skill_dataset_200.csv"
    try:
        df = pd.read_csv(csv_path)
        X = df[["addition", "subtraction", "multiplication", "division"]]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model, "trained_from_csv"
    except Exception as e:
        # ถ้าอ่านไฟล์ไม่สำเร็จ ให้สร้าง fallback model แบบง่าย ๆ
        class FallbackModel:
            def predict(self, X):
                # X เป็น list of lists: [[add, sub, mul, div]]
                out = []
                for row in X:
                    add, sub, mul, div = row
                    scores = {"add": add, "sub": sub, "mul": mul, "div": div}
                    min_score = min(scores.values())
                    # ถ้าทุกทักษะ >= 90 => strong_all
                    if all(v >= 90 for v in scores.values()):
                        out.append("strong_all")
                    else:
                        # ถ้ามีหลายจุดที่เท่ากัน เลือก 1 ตัวโดยเรียงลำดับ add, sub, mul, div (deterministic)
                        for k in ["add", "sub", "mul", "div"]:
                            if scores[k] == min_score:
                                out.append(f"weak_{k}")
                                break
                return out
        return FallbackModel(), f"fallback_no_csv ({e})"

model, model_source = train_model()

# =========================
# ฟังก์ชันสร้างโจทย์
# =========================
def generate_question(operation):
    if operation == "add":
        a, b = random.randint(1, 50), random.randint(1, 50)
        correct = a + b
        question = f"{a} + {b} = ?"

    elif operation == "sub":
        a, b = random.randint(1, 50), random.randint(1, 50)
        if a < b:
            a, b = b, a
        correct = a - b
        question = f"{a} - {b} = ?"

    elif operation == "mul":
        a, b = random.randint(1, 12), random.randint(1, 12)
        correct = a * b
        question = f"{a} × {b} = ?"

    elif operation == "div":
        b = random.randint(1, 12)
        correct = random.randint(1, 12)
        a = b * correct
        question = f"{a} ÷ {b} = ?"

    # สร้างตัวเลือกโดยหลีกเลี่ยงซ้ำกัน
    choices = set()
    choices.add(correct)
    while len(choices) < 4:
        delta = random.choice([1,2,3,4,5,6,7,8,9])
        sign = random.choice([1, -1])
        cand = correct + sign * delta
        if cand >= 0:
            choices.add(cand)
    choices = list(choices)
    random.shuffle(choices)

    return question, correct, choices

# =========================
# เตรียมข้อสอบ 12 ข้อ (เก็บใน session state)
# =========================
if "questions" not in st.session_state:
    st.session_state.questions = []
    operations = ["add"]*3 + ["sub"]*3 + ["mul"]*3 + ["div"]*3
    random.shuffle(operations)

    for op in operations:
        q, ans, choices = generate_question(op)
        st.session_state.questions.append({
            "operation": op,
            "question": q,
            "answer": ans,
            "choices": choices
        })

# =========================
# แสดงข้อสอบ
# =========================
scores = {"add":0, "sub":0, "mul":0, "div":0}

st.subheader("📘 ทำแบบทดสอบ 12 ข้อ")
st.caption("โปรดตอบทุกข้อ แล้วกดปุ่ม 'ส่งคำตอบ'")

PLACEHOLDER = "-- เลือกคำตอบ --"

if "user_answers" not in st.session_state or len(st.session_state.user_answers) != len(st.session_state.questions):
    st.session_state.user_answers = [PLACEHOLDER] * len(st.session_state.questions)

for i, q in enumerate(st.session_state.questions):
    choices_with_placeholder = [PLACEHOLDER] + [str(c) for c in q["choices"]]
    selected = st.radio(
        f"ข้อ {i+1}: {q['question']}",
        choices_with_placeholder,
        key=f"q{i}"
    )
    st.session_state.user_answers[i] = selected

# friendly names + video links map
friendly = {
    "add": ("การบวก", "https://www.youtube.com/watch?v=c5eS7nRsE_Q"),
    "sub": ("การลบ", "https://www.youtube.com/watch?v=vT_VBLlvdn8"),
    "mul": ("การคูณ", "https://www.youtube.com/watch?v=73obrcsERe8"),
    "div": ("การหาร", "https://www.youtube.com/watch?v=9D1JW8rYqeA")
}

# =========================
# ตรวจคำตอบเมื่อกดปุ่ม
# =========================
if st.button("ส่งคำตอบ"):
    if any(ans == PLACEHOLDER for ans in st.session_state.user_answers):
        st.warning("กรุณาตอบให้ครบทุกข้อก่อนส่ง (ยังมีข้อที่ไม่ได้เลือกคำตอบ).")
    else:
        # คำนวณคะแนน
        for i, q in enumerate(st.session_state.questions):
            user_val = st.session_state.user_answers[i]
            try:
                user_val_num = int(user_val)
            except:
                user_val_num = None
            if user_val_num == q["answer"]:
                scores[q["operation"]] += 1

        # Normalize 0–100 (แต่ละหมวดมี 3 ข้อ)
        add_score = round((scores["add"]/3)*100, 2)
        sub_score = round((scores["sub"]/3)*100, 2)
        mul_score = round((scores["mul"]/3)*100, 2)
        div_score = round((scores["div"]/3)*100, 2)

        st.subheader("📊 ผลคะแนน")
        st.write(f"➕ การบวก: {add_score}")
        st.write(f"➖ การลบ: {sub_score}")
        st.write(f"✖ การคูณ: {mul_score}")
        st.write(f"➗ การหาร: {div_score}")

        # ถ้าได้เต็มทุกหมวด
        if add_score == 100 and sub_score == 100 and mul_score == 100 and div_score == 100:
            st.success("🎉 คุณพร้อมเรียนบทต่อไปแล้ว!")
        else:
            # ส่งเข้า ML (หรือ fallback)
            try:
                prediction = model.predict([[add_score, sub_score, mul_score, div_score]])
                result = prediction[0]
            except Exception as e:
                result = "strong_all"  # fallback safe

            st.subheader("🤖 ผลการวิเคราะห์จาก AI (model prediction)")
            st.info(f"โมเดล ({model_source}) ทำนายจุดที่ควรพัฒนา: {result}")

            # --- วิเคราะห์จากคะแนนจริง (tie-aware) ---
            skill_scores = {
                "add": add_score,
                "sub": sub_score,
                "mul": mul_score,
                "div": div_score
            }

            min_score = min(skill_scores.values())
            weakest = [k for k, v in skill_scores.items() if v == min_score]

            st.subheader("🔎 วิเคราะห์จากคะแนนจริง (tie-aware)")
            st.write(f"คะแนนต่ำสุดคือ {min_score} — หัวข้อที่คะแนนต่ำสุด (จุดอ่อน):")

            for w in weakest:
                name, vid = friendly[w]
                st.write(f"- {name} (คะแนน {skill_scores[w]} / 100)")
                st.write(f"  → คำแนะนำ: ฝึก{ name } เพิ่มเติม")
                st.video(vid)

            # ถ้าโมเดลแนะนำหมวดอื่น ให้แสดงเสริม
            # แปลง label model เป็น key ถ้ามี
            model_map = {
                "weak_add": "add",
                "weak_sub": "sub",
                "weak_mul": "mul",
                "weak_div": "div"
            }
            if result not in [f"weak_{w}" for w in weakest] and result != "strong_all":
                st.write("")  # spacer
                st.write("หมายเหตุ: โมเดลยังชี้ไปที่:", result)
                if result in model_map:
                    mm = model_map[result]
                    st.write(f"โมเดลแนะนำให้ฝึก {friendly[mm][0]} ด้วย (เสริม)")
                    st.video(friendly[mm][1])

            # -------------------------
# Improved: สร้างกราฟ Radar (Spider) — tie-aware, ปรับตำแหน่งป้ายไม่ให้ทับ
# -------------------------
import numpy as np
import matplotlib.pyplot as plt

# ป้ายแสดงบนแกน (แสดงเป็นภาษาไทย)
labels = ["การบวก", "การลบ", "การคูณ", "การหาร"]
# ค่า (แน่ใจว่าเป็นตัวเลข)
values = [float(add_score), float(sub_score), float(mul_score), float(div_score)]

# จำนวนแกน
N = len(labels)

# มุมสำหรับแต่ละแกน (ไม่รวม endpoint) แล้วเติมจุดแรกตอนปิดวง
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# เติมค่าซ้ำจุดแรกเพื่อปิดวง
vals = values + values[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

# เริ่มที่มุมบน (90 deg) และหมุนตามเข็มนาฬิกา ให้ดูเป็น radar แบบทั่วไป
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# เส้นกราฟและพื้นที่ด้านใน
ax.plot(angles, vals, linewidth=2)
ax.fill(angles, vals, alpha=0.25)

# ตั้งชื่อแกน (theta labels)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)

# ขอบเขตรัศมี 0-100 และ tick ที่อ่านง่าย
ax.set_ylim(0, 100)
ax.set_rlabel_position(180 / N)  # ตำแหน่ง radial labels (ปรับถ้าต้องการ)
ax.yaxis.set_ticks([20, 40, 60, 80, 100])
ax.yaxis.set_ticklabels(['20','40','60','80','100'])

# เส้นตารางให้เห็นชัด
ax.xaxis.grid(True, linestyle='-', linewidth=0.5)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5)

# ไฮไลต์ทุกจุดที่เป็นคะแนนต่ำสุด (tie-aware)
min_score = min(values)
min_indices = [i for i, v in enumerate(values) if v == min_score]

for mi in min_indices:
    ang = angles[mi]
    val = values[mi]
    # จุดแดง/ขอบดำ
    ax.plot(ang, val, marker='o', markersize=10, markeredgecolor='k', markerfacecolor='red')
    # ตำแหน่งข้อความเล็กน้อย (offset) เพื่อไม่ให้ทับจุด
    # กำหนด alignment ตามตำแหน่งเชิงมุม (cos ใช้เช็คซ้าย/ขวา)
    ha = 'center'
    if np.cos(ang) < -0.2:
        ha = 'right'
    elif np.cos(ang) > 0.2:
        ha = 'left'
    # วางข้อความเป็น offset point (ยกขึ้นเล็กน้อย)
    ax.annotate(f"{labels[mi]}: {values[mi]:.1f}", xy=(ang, val),
                xytext=(0, 12), textcoords='offset points',
                ha=ha, va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

# หัวข้อ
ax.set_title("กราฟ Radar: คะแนนทักษะ (0–100)", pad=20)

# แสดงใน Streamlit
st.pyplot(fig)

# =========================
# ปุ่มเริ่มใหม่
# =========================
if st.button("🔄 เริ่มใหม่"):
    keys_to_clear = ["questions", "user_answers"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.script_request_rerun()
        except Exception:
            st.stop()
