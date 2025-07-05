import os
import json
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import glob

# Set up API
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# load config
with open("C:/Users/ADMIN/OneDrive/Tài liệu/Vidu/mindx_python/csi/endproject/web/pages/config.json", "r",encoding="utf-8") as f:
    config = json.load(f)
    functions = config.get("functions", 'Gioi thiệu về tiền ảo')
    initial_bot_message = config.get("initial_bot_message", "hello i am chatbot?")

# Load data
data = '../data/coin_*.csv'
csv_files = glob.glob(data)
dfs = [pd.read_csv(f) for f in csv_files]
all_coins = pd.concat(dfs, ignore_index=True)

all_coins = pd.DataFrame(all_coins)
adfd = pd.read_csv('../coin_history.csv', on_bad_lines='skip')

asd = list(all_coins['Name'])
asd = list(dict.fromkeys(asd))
# create LLM
model = genai.GenerativeModel("gemini-1.5-flash",
                              system_instruction=f"""
                              Bạn tên là CryAI, một trợ lý AI có nhiệm vụ hỗ trợ giải đáp thông tin cho khách hàng về tiền ảo.
                              
                              Các chức năng mà bạn hỗ trợ gồm:
                              bạn có nguồn dữ liệu này {all_coins}
                              1.Gioi thiệu về tiền ảo tu file {adfd}
                              2.Gioi thiệu các loại tiền ảo phổ biến nhất qua tên {asd}
                              3 .Hướng dẫn cách đầu tư vào tiền ảo
                              Đối với các câu hỏi ngoài chức năng mà bạn hỗ trợ, trả lời bằng 'Tôi đang không hỗ trợ chức năng này. 
                              """)

# ui
st.title("CryAI - Trợ lý AI về Tiền ảo")
st.write("Chào mừng bạn đến với CryAI, trợ lý AI của chúng tôi. Bạn có thể hỏi tôi về các chức năng sau:")
st.write("- Giới thiệu về tiền ảo")
st.write("- Giới thiệu các loại tiền ảo phổ biến nhất")
st.write("- Hướng dẫn cách đầu tư vào tiền ảo")
try:
    prompt = st.text_input("Bạn có câu hỏi gì không?", placeholder="Nhập câu hỏi của bạn tại đây...")
    if prompt:
        response = model.generate_content(prompt)
        st.write(response.text)
except Exception as e:
    st.error("Đã xảy ra lỗi. Vui lòng thử lại sau.")
    
    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    st.write("## Lịch sử trò chuyện")
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Bạn:** {user_msg}")
        st.markdown(f"**CryAI:** {bot_msg}")

    # Add new messages to history
    if prompt:
        st.session_state.chat_history.append((prompt, response.text))