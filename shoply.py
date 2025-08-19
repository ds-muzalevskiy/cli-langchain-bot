import os
import json
import uuid
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage
from pathlib import Path
from google.colab import userdata
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL")
BRAND = os.getenv("BRAND_NAME")

with open("data/faq.json", encoding="utf-8") as f:
    FAQ = json.load(f)

with open("data/orders.json", encoding="utf-8") as f:
    ORDERS = json.load(f)

llm = ChatOpenAI(model=MODEL, openai_api_key=API_KEY, temperature=0)
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory, verbose=False)

Path("logs").mkdir(exist_ok=True)
session_id = uuid.uuid4().hex
log_file = f"logs/session_{session_id}.jsonl"
token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def save_log(role, content):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "role": role,
        "content": content
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def count_tokens(input_text, output_text):
    input_tokens = len(input_text.split())
    output_tokens = len(output_text.split())
    token_usage["prompt_tokens"] += input_tokens
    token_usage["completion_tokens"] += output_tokens
    token_usage["total_tokens"] += input_tokens + output_tokens

def find_faq_answer(question):
    for pair in FAQ:
        if pair["q"].lower() in question.lower():
            return pair["a"]
    return None

def get_order_status(order_id):
    order = ORDERS.get(order_id)
    if not order:
        return f"Извините, заказ с номером {order_id} не найден."
    status = order.get("status")
    if status == "in_transit":
        return f"Ваш заказ {order_id} в пути через {order['carrier']}, ожидается через {order['eta_days']} дней."
    elif status == "delivered":
        return f"Заказ {order_id} был доставлен {order['delivered_at']}."
    elif status == "processing":
        return f"Заказ {order_id} обрабатывается: {order.get('note', 'дополнительной информации нет')}."
    else:
        return f"Статус заказа {order_id}: {status}"

def main():
    print(f"Добро пожаловать в поддержку {BRAND}!")
    print("Введите ваш вопрос или команду. Для выхода напишите `exit`.\n")

    while True:
        user_input = input("🧑 Вы: ")
        if user_input.strip().lower() in ["exit", "выход"]:
            break

        save_log("user", user_input)

        if user_input.startswith("/order"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                order_reply = get_order_status(parts[1])
                print("Бот:", order_reply)
                save_log("bot", order_reply)
                count_tokens(user_input, order_reply)
                continue
            else:
                reply = "Пожалуйста, используйте команду в формате: `/order <id>`."
                print("Бот:", reply)
                save_log("bot", reply)
                continue

        faq_answer = find_faq_answer(user_input)
        if faq_answer:
            print("Бот:", faq_answer)
            save_log("bot", faq_answer)
            count_tokens(user_input, faq_answer)
        else:
            prompt = f"Ты — вежливый помощник бренда {BRAND}. Отвечай кратко и по делу. Если не уверен — не придумывай. Вопрос: {user_input}"
            bot_reply = chain.run(prompt)            
            print("Бот:", bot_reply)
            save_log("bot", bot_reply)
            count_tokens(user_input, bot_reply)

    print("\n Сессия завершена.")
    print("Токены:", token_usage)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"usage": token_usage}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()