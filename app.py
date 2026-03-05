# app.py (แก้ไขแล้ว)
import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Math Skill AI", layout="centered")
st.title("🧠 ระบบวิเคราะห์จุดอ่อนคณิตศาสตร์ด้วย AI + Radar Chart (แก้ป้ายทับ)")

# =========================
# Train model (with fallback)
# =========================
@st.cache_resource
def train_model(csv_path="math_skill_dataset_200.csv"):
    try:
        df = pd.read_csv(csv_path)
        required_cols = {"addition", "subtraction", "multiplication", "division", "label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{csv_path} missing required columns: {required_cols - set(df.columns)}")

        X = df[["addition", "subtraction", "multiplication", "division"]]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model, "trained_from_csv"
    except Exception as e:
        class FallbackModel:
            def predict(self, X):
                out = []
                for row in X:
                    try:
                        add, sub, mul, div = [float(x) for x in row]
                    except Exception:
                        add = sub = mul = div = 0.0
                    scores = {"add": add, "sub": sub, "mul": mul, "div": div}
                    if all(v >= 90 for v in scores.values()):
                        out.append("strong_all")
                    else:
                        min_val = min(scores.values())
                        for k in ["add", "sub", "mul", "div"]:
                            if scores[k] == min_val:
                                out.append(f"weak_{k}")
                                break
                return out
        return FallbackModel(), f"fallback_no_csv ({e})"

model, model_source = train_model()

# =========================
# Utility: generate question
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
    else:
        raise ValueError("Unknown operation")

    choices = set([correct])
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
# Prepare questions in session_state
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

PLACEHOLDER = "-- เลือกคำตอบ --"
if "user_answers" not in st.session_state or len(st.session_state.user_answers) != len(st.session_state.questions):
    st.session_state.user_answers = [PLACEHOLDER] * len(st.session_state.questions)

friendly = {
    "add": ("การบวก", "https://www.youtube.com/watch?v=c5eS7nRsE_Q"),
    "sub": ("การลบ", "https://www.youtube.com/watch?v=vT_VBLlvdn8"),
    "mul": ("การคูณ", "https://www.youtube.com/watch?v=73obrcsERe8"),
    "div": ("การหาร", "https://www.youtube.com/watch?v=9D1JW8rYqeA")
}

# =========================
# SHOW QUIZ
# =========================
st.subheader("📘 ทำแบบทดสอบ 12 ข้อ")
st.caption("โปรดตอบทุกข้อ แล้วกดปุ่ม 'ส่งคำตอบ'")

for i, q in enumerate(st.session_state.questions):
    choices_with_placeholder = [PLACEHOLDER] + [str(c) for c in q["choices"]]
    selected = st.radio(f"ข้อ {i+1}: {q['question']}", choices_with_placeholder, key=f"q{i}")
    st.session_state.user_answers[i] = selected

# =========================
# Improved Radar plot function (fix overlapping labels)
# =========================
def make_radar_figure(values, labels):
    """
    values: list of 4 numeric values (0-100)
    labels: list of 4 strings
    """
    vals = [float(v) if v is not None else 0.0 for v in values]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals_loop = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    # start at top and clockwise
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # draw plot and fill
    ax.plot(angles, vals_loop, linewidth=2)
    ax.fill(angles, vals_loop, alpha=0.25)

    # labels and radial limits
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 100)
    ax.yaxis.set_ticks([20,40,60,80,100])
    ax.yaxis.set_ticklabels(['20','40','60','80','100'])

    # move radial labels a bit to avoid overlapping center annotation area
    # choose an angle where radial labels are placed (in degrees)
    ax.set_rlabel_position(225)

    ax.xaxis.grid(True, linestyle='-', linewidth=0.5)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5)

    # If all values are zero or very small, we still want readable annotations:
    # We'll compute an outward offset in "offset points" based on angle and ensure min offset.
    min_score = min(vals) if len(vals) > 0 else 0.0
    min_indices = [i for i, v in enumerate(vals) if v == min_score]

    for mi in min_indices:
        ang = angles[mi]
        val = vals[mi]

        # marker
        ax.plot(ang, val, marker='o', markersize=10, markeredgecolor='k', markerfacecolor='red', zorder=5)

        # compute offset direction (points): push text radially outward from the point
        # use a base radial offset (points) larger when value is small
        base_outward = 18  # base distance in offset points
        extra = 0
        if val < 10:
            extra = 14
        elif val < 30:
            extra = 8
        # offset_x, offset_y in offset points (mathematical x,y)
        offset_x = (base_outward + extra) * np.cos(ang)
        offset_y = (base_outward + extra) * np.sin(ang)

        # determine alignment based on angle
        ha = 'center'
        if np.cos(ang) < -0.3:
            ha = 'right'
        elif np.cos(ang) > 0.3:
            ha = 'left'
        va = 'center'
        if np.sin(ang) < -0.3:
            va = 'top'
        elif np.sin(ang) > 0.3:
            va = 'bottom'

        # Annotate using offset points so text moves away from center
        ax.annotate(f"{labels[mi]}: {val:.1f}", xy=(ang, val),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    ha=ha, va=va, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8), zorder=6)

    ax.set_title("กราฟ Radar: คะแนนทักษะ (0–100)", pad=20)
    plt.tight_layout()
    return fig

# =========================
# When user clicks "ส่งคำตอบ"
# =========================
if st.button("ส่งคำตอบ"):
    if any(ans == PLACEHOLDER for ans in st.session_state.user_answers):
        st.warning("กรุณาตอบให้ครบทุกข้อก่อนส่ง (ยังมีข้อที่ไม่ได้เลือกคำตอบ).")
    else:
        tally = {"add": 0, "sub": 0, "mul": 0, "div": 0}
        for i, q in enumerate(st.session_state.questions):
            user_val = st.session_state.user_answers[i]
            try:
                user_val_num = int(user_val)
            except Exception:
                user_val_num = None
            if user_val_num == q["answer"]:
                tally[q["operation"]] += 1

        add_score = round((tally["add"]/3)*100, 2)
        sub_score = round((tally["sub"]/3)*100, 2)
        mul_score = round((tally["mul"]/3)*100, 2)
        div_score = round((tally["div"]/3)*100, 2)

        st.session_state['latest_scores'] = {
            "add": add_score,
            "sub": sub_score,
            "mul": mul_score,
            "div": div_score
        }

        st.subheader("📊 ผลคะแนน")
        st.write(f"➕ การบวก: {add_score}")
        st.write(f"➖ การลบ: {sub_score}")
        st.write(f"✖ การคูณ: {mul_score}")
        st.write(f"➗ การหาร: {div_score}")

        if add_score == 100 and sub_score == 100 and mul_score == 100 and div_score == 100:
            st.success("🎉 คุณพร้อมเรียนบทต่อไปแล้ว!")
        else:
            try:
                prediction = model.predict([[add_score, sub_score, mul_score, div_score]])
                result = prediction[0]
            except Exception:
                result = "strong_all"

            st.subheader("🤖 ผลการวิเคราะห์จาก AI (model prediction)")
            st.info(f"โมเดล ({model_source}) ทำนายจุดที่ควรพัฒนา: {result}")

            skill_scores = {"add": add_score, "sub": sub_score, "mul": mul_score, "div": div_score}
            min_score = min(skill_scores.values())
            weakest = [k for k, v in skill_scores.items() if v == min_score]

            st.subheader("🔎 วิเคราะห์จากคะแนนจริง (tie-aware)")
            st.write(f"คะแนนต่ำสุดคือ {min_score} — หัวข้อที่คะแนนต่ำสุด (จุดอ่อน):")
            for w in weakest:
                name, vid = friendly[w]
                st.write(f"- {name} (คะแนน {skill_scores[w]} / 100)")
                st.write(f"  → คำแนะนำ: ฝึก{name} เพิ่มเติม")
                st.video(vid)

            model_map = {"weak_add":"add","weak_sub":"sub","weak_mul":"mul","weak_div":"div"}
            if result not in [f"weak_{w}" for w in weakest] and result != "strong_all":
                st.write("")
                st.write("หมายเหตุ: โมเดลยังชี้ไปที่:", result)
                if result in model_map:
                    mm = model_map[result]
                    st.write(f"โมเดลแนะนำให้ฝึก {friendly[mm][0]} ด้วย (เสริม)")
                    st.video(friendly[mm][1])

# =========================
# Show radar chart if we have latest_scores in session_state
# =========================
if 'latest_scores' in st.session_state:
    scores_dict = st.session_state['latest_scores']
    labels = ["การบวก", "การลบ", "การคูณ", "การหาร"]
    vals = [
        float(scores_dict.get("add", 0.0)),
        float(scores_dict.get("sub", 0.0)),
        float(scores_dict.get("mul", 0.0)),
        float(scores_dict.get("div", 0.0))
    ]
    fig = make_radar_figure(vals, labels)
    st.pyplot(fig)
else:
    st.info("กราฟ Radar จะแสดงหลังจากทำแบบทดสอบแล้วกด 'ส่งคำตอบ'")

# =========================
# Reset button
# =========================
st.markdown("---")
if st.button("🔄 เริ่มใหม่"):
    keys_to_clear = ["questions", "user_answers", "latest_scores"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()
