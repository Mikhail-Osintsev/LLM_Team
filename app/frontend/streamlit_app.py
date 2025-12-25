import os                                      # окружение
import requests                                # HTTP-запросы к бэкенду
import streamlit as st                         # Streamlit UI
from datetime import datetime

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")  # адрес API

st.set_page_config(page_title="Book RAG", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    .stMarkdown a svg { display: none; }
    [data-testid="stHeaderActionElements"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.title("Book RAG — поиск ответов по книге")

# Инициализация session_state
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "selected_books" not in st.session_state:
    st.session_state.selected_books = []

# Загружаем список книг
@st.cache_data(ttl=300)
def get_books():
    try:
        r = requests.get(f"{BACKEND_URL}/books", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

books = get_books()

# Выбор книг
if books:
    st.markdown("### Выберите книги для поиска")

    # Создаем grid для отображения книг
    cols_per_row = 3
    for i in range(0, len(books), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(books):
                book = books[i + j]
                with col:
                    # Создаем карточку книги
                    is_selected = book["filename"] in st.session_state.selected_books

                    button_label = f"{'✓ ' if is_selected else ''}{book['title']}"
                    button_type = "primary" if is_selected else "secondary"

                    if st.button(
                        button_label,
                        key=f"book_{book['filename']}",
                        use_container_width=True,
                        type=button_type
                    ):
                        # Переключаем выбор книги
                        if book["filename"] in st.session_state.selected_books:
                            st.session_state.selected_books.remove(book["filename"])
                        else:
                            st.session_state.selected_books.append(book["filename"])
                        st.rerun()

    st.markdown("---")

# Основной интерфейс поиска
col1, col2 = st.columns([4, 1])
with col1:
    q = st.text_input("Ваш вопрос", key="question_input", label_visibility="visible")
with col2:
    top_k = st.selectbox("Фрагментов", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=3, label_visibility="visible")

# Обработка нажатия Enter или кнопки "Спросить"
search_triggered = st.button("Спросить", use_container_width=True, type="primary")

if search_triggered:
    if not q.strip():
        st.warning("Введите вопрос")
    else:
        with st.spinner("Ищу ответ..."):
            r = requests.post(
                f"{BACKEND_URL}/ask", json={"question": q, "top_k": top_k}, timeout=60
            )
            if r.status_code != 200:
                st.error(f"Ошибка API: {r.text}")
            else:
                data = r.json()

                # Сохраняем результат в историю
                search_result = {
                    "question": q,
                    "answer": data["answer"],
                    "passages": data["passages"],
                    "top_k": top_k,
                    "timestamp": datetime.now()
                }

                # Добавляем в начало списка и оставляем только последние 10
                st.session_state.search_history.insert(0, search_result)
                st.session_state.search_history = st.session_state.search_history[:10]

                # Отображаем результат
                st.markdown("### Ответ")
                st.write(data["answer"])

                st.markdown("### Цитаты")
                for i, passage in enumerate(data["passages"], start=1):
                    # Формируем заголовок цитаты с метаданными
                    metadata = passage.get("metadata", {})
                    book_name = metadata.get("book_name", "")
                    page_number = metadata.get("page_number", 0)
                    score = passage.get("score", 0.0)

                    # Создаем информативный заголовок
                    header_parts = [f"[{i}] score={score:.3f}"]
                    if book_name and page_number:
                        header_parts.append(f"— {book_name}, стр. {page_number}")
                    elif book_name:
                        header_parts.append(f"— {book_name}")

                    with st.expander(" ".join(header_parts)):
                        st.write(passage.get("text", ""))

                        # Опционально: добавим ссылку для открытия PDF (пока заглушка)
                        if page_number and metadata.get("filename"):
                            st.caption(f"📄 Источник: {metadata.get('filename')}, страница {page_number}")

# Блок "Последние 10 запросов"
if st.session_state.search_history:
    st.markdown("---")
    st.markdown("### Ваши последние 10 запросов")

    for idx, item in enumerate(st.session_state.search_history):
        timestamp_str = item["timestamp"].strftime("%d.%m.%Y %H:%M")

        with st.expander(f"{timestamp_str} — {item['question'][:60]}{'...' if len(item['question']) > 60 else ''}"):
            st.markdown(f"**Вопрос:** {item['question']}")
            st.markdown(f"**Ответ:** {item['answer']}")
            st.markdown("**Цитаты:**")
            for i, passage in enumerate(item["passages"], start=1):
                # Поддержка как нового формата (dict), так и старого (tuple)
                if isinstance(passage, dict):
                    metadata = passage.get("metadata", {})
                    score = passage.get("score", 0.0)
                    txt = passage.get("text", "")
                    book_name = metadata.get("book_name", "")
                    page_number = metadata.get("page_number", 0)

                    info = f"[{i}] score={score:.3f}"
                    if book_name and page_number:
                        info += f" — {book_name}, стр. {page_number}"
                    st.text(info)
                else:
                    # Старый формат (tuple)
                    txt, score = passage
                    st.text(f"[{i}] score={score:.3f}")

                st.caption(txt[:200] + "..." if len(txt) > 200 else txt)