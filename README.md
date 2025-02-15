# 🚀 IndianTaxGPT: AI-Powered Tax Simplification for India

**IndianTaxGPT** is a cutting-edge AI-powered platform designed to demystify Indian tax laws and empower users with clear, actionable, and reliable information. By integrating state-of-the-art technologies like NLP, GPT-powered models, and advanced vector search, IndianTaxGPT serves as a virtual tax consultant—bridging the gap between complex tax regulations and everyday users.

---

## ⚡ The Challenge: Simplifying Complexity

Navigating the labyrinth of Indian tax laws can be overwhelming:

- **Information Overload:** Individuals and enterprises grapple with convoluted regulations.
- **Missed Opportunities:** Missteps lead to errors, missed deductions, and potential penalties.
- **Accessibility Barriers:** Limited access to affordable and accurate tax consultancy.

---

## 💡 Our Vision: Tax Compliance, Made Effortless

IndianTaxGPT transforms the tax management experience for individuals and businesses alike. Here's how:

- **AI-Powered Simplicity:** Leveraging GPT-driven intelligence for intuitive and accurate guidance.
- **Comprehensive Insights:** Covering tax laws, filing processes, deductions, exemptions, and beyond.
- **Empowering Accessibility:** User-centric design for effortless navigation and decision-making.

---

## 🛠️ Technologies at the Core

- **🌐 Natural Language Processing (NLP):** Advanced text processing for precise contextual understanding.
- **🤖 GPT-Powered Models:** Llama 2-7b V delivers robust AI-driven tax insights.
- **📈 Embedding Search Database:** Huggingface Transformers + Pinecone Vector DB enable lightning-fast data retrieval.
- **🔗 Langchain LLM Integration:** Seamlessly orchestrating modular AI components.
- **💻 Streamlit UI:** A clean, interactive web interface for real-time tax assistance.

---

## 🧩 Technical Architecture

IndianTaxGPT utilizes **Pinecone VectorDB** to store tax data embeddings. Here's how it works:

1. **User Query** → Transformed into embeddings via NLP models.
2. **Vector Search** → Compared against database embeddings for relevant matches.
3. **GPT Integration** → Processes matched embeddings to generate clear, actionable answers.

> **Models in Action:** Open-source LLMs and embeddings balance cost and performance for Indian tax intricacies.

---

## 🔍 Challenges We Conquered

1. **Cost Efficiency:** Leveraged open-source models to bypass expensive APIs.
2. **Performance Trade-offs:** Balanced computational speed with affordability.
3. **Deployment Constraints:** Transitioned to local execution with modular and replicable architecture.

---

## 🚀 How to Get Started

1. Clone the repository:
   ```bash
   git clone https://github.com/NitinReddy-dev/IncomeTAXGPT.git
   cd IncomeTAXGPT
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r models/requirements.txt
   ```
3. Add your data: Place tax-related PDFs in the `/data` folder.
4. Set up Pinecone: [Create an account](https://www.pinecone.io) and configure your index.
5. Embed data: Run `embedder.py` with your Pinecone API keys and index name:
   ```bash
   python embedder.py
   ```
6. Launch the app: Start `app.py` with your API keys and index name:
   ```bash
   python app.py
   ```

---

## 🖼️ Screenshots

| **Interface** | **User Interaction** |
|---------------|----------------------|
| ![Screenshot1](https://github.com/NitinReddy-dev/IncomeTAXGPT/assets/114919978/a0666954-e1e2-4949-badd-217f3058db0f) | ![Screenshot2](https://github.com/NitinReddy-dev/IncomeTAXGPT/assets/114919978/867cc00d-a045-4de5-9736-a81ee5c4d4de) |

---

## 📊 Features

- **Custom Embedding Pipeline:** Tax-related PDFs are converted into searchable embeddings.
- **Interactive Query Handling:** Handles complex tax queries with conversational accuracy.
- **Cost-Effective Setup:** Designed with open-source tools to minimize costs without compromising functionality.
- **User-Centric UI:** Simple and intuitive interface built using Streamlit.

---

## 🤝 Contribute to IndianTaxGPT

We’re on a mission to simplify taxes in India. Your contributions—whether through feedback, code, or spreading the word—are invaluable.

- **🔗 Repo:** [IndianTaxGPT on GitHub](https://github.com/NitinReddy-dev/IncomeTAXGPT)
- **📬 Contact:** [Your email/Discord/etc.]

---

**💡 Together, let’s make tax compliance smarter and simpler!**
