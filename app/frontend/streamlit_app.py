import os                                      # окружение
import requests                                # HTTP-запросы к бэкенду
import streamlit as st                         # Streamlit UI

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")  # адрес API

st.set_page_config(page_title="Book RAG", layout="wide")   # заголовок/ширина
st.title("Book RAG — поиск ответов по книге")              # заголовок в UI

q = st.text_input("Ваш вопрос")                            # поле ввода
top_k = st.slider("Сколько фрагментов искать", 1, 10, 4)   # слайдер k

if st.button("Спросить"):                                   # кнопка
    if not q.strip():                                       # пустой ввод
        st.warning("Введите вопрос")                        # предупреждение
    else:
        with st.spinner("Ищу ответ..."):                    # индикатор
            r = requests.post(                              # POST /ask
                f"{BACKEND_URL}/ask", json={"question": q, "top_k": top_k}, timeout=60
            )
            if r.status_code != 200:                        # если ошибка
                st.error(f"Ошибка API: {r.text}")           # показываем текст
            else:
                data = r.json()                             # JSON-ответ
                st.subheader("Ответ")                       # подзаголовок
                st.write(data["answer"])                    # текст ответа
                st.subheader("Цитаты")                      # подзаголовок
                for i, (txt, score) in enumerate(data["passages"], start=1):  # перечисляем
                    with st.expander(f"[{i}] score={score:.3f}"):             # разворачиваемый блок
                        st.write(txt)                         # показываем пассаж