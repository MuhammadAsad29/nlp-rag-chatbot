from google import genai
from google.genai import types
from google.genai.errors import ClientError

import os
from rag.retriever import Retriever
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# gemini-2.5-flash has 1,500 RPD free tier vs gemini-3.6-flash which has only 20 RPD free tier
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.1-flash-lite"

class RAGChatbot:
    def __init__(self, documents):
        self.retriever = Retriever(documents)

    def ask(self, question, top_k=5, temperature=0.3):
        context_chunks = self.retriever.retrieve(question, top_k=top_k)
        context_str = "\n---\n".join(context_chunks)

        prompt = f"""
You are an expert AI assistant specializing in NLP and RAG architectures.
Answer the user's question using the retrieved context below.

Instructions:
1. Synthesize a clear, natural, and comprehensive response based on the context.
2. Do not just copy-paste snippets; explain the concepts clearly.
3. If the context does not contain enough information to answer the question accurately, state what you know from context and note that details are limited.

Retrieved Context:
{context_str}

User Question:
{question}

Answer:
"""

        try:
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature
                )
            )
            return response.text, context_chunks
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                try:
                    response = client.models.generate_content(
                        model=FALLBACK_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=temperature
                        )
                    )
                    return response.text, context_chunks
                except Exception as fallback_err:
                    return f"⚠️ **Rate Limit Reached**: Free tier quota temporarily exceeded. Please wait a few seconds before asking another question.\n\n*(Detail: {fallback_err})*", context_chunks
            return f"⚠️ **API Error**: {e}", context_chunks
        except Exception as e:
            return f"⚠️ **Unexpected Error**: {e}", context_chunks


