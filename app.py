import gradio as gr
from rag.chatbot import RAGChatbot

def load_and_chunk_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

# Initialize RAG System
print("⏳ Initializing Vector Index & Loading Knowledge Base...")
docs = load_and_chunk_documents("data/Diverse_NLP_QA.txt")
bot = RAGChatbot(docs)
print("✅ RAG System Ready!")

# Custom CSS for rich aesthetics
custom_css = """
#main-container { max-width: 1200px; margin: 0 auto; }
.header-badge {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 8px;
}
.context-box {
    background-color: rgba(99, 102, 241, 0.05);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 8px;
    padding: 12px;
    font-size: 0.85rem;
}
"""

def user_submit(user_message, history, top_k, temperature):
    if not user_message.strip():
        return "", history, "No context retrieved yet."
    
    history = history or []
    answer, context_chunks = bot.ask(user_message, top_k=int(top_k), temperature=float(temperature))
    
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    
    formatted_context = "### 🔍 FAISS Vector Retrieved Context:\n\n"
    for i, chunk in enumerate(context_chunks, 1):
        formatted_context += f"**[Chunk {i}]**\n{chunk}\n\n---\n"
    
    return "", history, formatted_context


with gr.Blocks(title="NLP RAG Chatbot") as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown(
            """
            <div style='text-align: center; margin-bottom: 1rem;'>
                <span class='header-badge'>Gemini 3.5 Flash lite + FAISS Vector Search</span>
                <h1 style='font-size: 2.2rem; margin: 0.2rem 0;'>🤖 NLP & RAG Intelligent Assistant</h1>
                <p style='color: #6b7280;'>Ask any question about Natural Language Processing, Vector Search, and RAG Architecture.</p>
            </div>
            """
        )
        
        with gr.Row():
            # Main Chat Column
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    height=500
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about NLP, RAG, Embeddings, Tokenization...",
                        show_label=False,
                        scale=8,
                        container=False
                    )
                    send_btn = gr.Button("Send 🚀", variant="primary", scale=2)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
                
                gr.Examples(
                    examples=[
                        "What is RAG and how does it work?",
                        "What is corpus in NLP?",
                        "Why is recall important in QA systems?",
                        "What is precision-recall tradeoff?",
                        "What is document retrieval?"
                    ],
                    inputs=msg_input,
                    label="💡 Example Questions"
                )
            
            # RAG Inspector & Controls Column
            with gr.Column(scale=5):
                with gr.Accordion("⚙️ RAG Hyperparameters", open=True):
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top-K Retrieved Context Chunks",
                        info="Number of nearest vector chunks to pull from FAISS"
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Temperature",
                        info="Lower values (0.2-0.4) ensure strict factual grounding"
                    )
                
                with gr.Accordion("📚 Live RAG Context Inspector", open=True):
                    context_inspector = gr.Markdown(
                        value="*Ask a question to inspect the raw context retrieved from FAISS vector index.*",
                        elem_classes=["context-box"]
                    )

    # Event Handlers
    msg_input.submit(
        user_submit,
        inputs=[msg_input, chatbot, top_k_slider, temp_slider],
        outputs=[msg_input, chatbot, context_inspector]
    )
    send_btn.click(
        user_submit,
        inputs=[msg_input, chatbot, top_k_slider, temp_slider],
        outputs=[msg_input, chatbot, context_inspector]
    )
    clear_btn.click(lambda: ([], "*Context cleared.*"), None, outputs=[chatbot, context_inspector])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
        css=custom_css,
        share=False
    )



