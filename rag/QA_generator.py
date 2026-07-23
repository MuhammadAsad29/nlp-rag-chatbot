from google import genai
import os
import time
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.6-flash"

OUTPUT_FILE = "D:\\PY Projects\\NLP RAG chatbot\\data\\Diverse_NLP_QA.txt"
TOPICS_FILE = "D:\\PY Projects\\NLP RAG chatbot\\data\\topics.txt"

QA_PER_BATCH = 20       # 🔑 SAFE SIZE
SLEEP_BETWEEN_CALLS = 35  # 🔑 Avoid 429

def load_topics():
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        return [t.strip() for t in f if t.strip()]

def generate_qa(topic, start_index, count):
    prompt = f"""
Generate {count} UNIQUE question-answer pairs about:

Topic: {topic}

Rules:
- No repeated ideas
- No paraphrased duplicates
- One concept per question
- Clear factual answers
- STRICT FORMAT ONLY:

Q{start_index}: question
A{start_index}: answer

Continue numbering sequentially.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text.strip()


def append(text):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n\n")

def main():
    topics = load_topics()
    qa_index = 401

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    for topic in topics:
        print(f"\n📌 Topic: {topic}")

        for _ in range(8):  # 8 × 25 = 200 per topic
            try:
                print(f"Generating QA {qa_index} → {qa_index + QA_PER_BATCH - 1}")
                qa_text = generate_qa(topic, qa_index, QA_PER_BATCH)
                append(qa_text)
                qa_index += QA_PER_BATCH
                time.sleep(SLEEP_BETWEEN_CALLS)

                if qa_index > 1000:
                    print("✅ 5000 QA reached")
                    return

            except Exception as e:
                print("⚠️ Rate limit hit, waiting 60s...")
                time.sleep(60)

    print("✅ Dataset generation completed")

if __name__ == "__main__":
    main()
