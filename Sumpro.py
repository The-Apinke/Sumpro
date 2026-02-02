import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os, io, fitz, re, time, hashlib
from pathlib import Path

# Load .env from the same directory as this script
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(f"OPENAI_API_KEY not found! Checked: {env_path}")
os.environ['OPENAI_API_KEY'] = api_key

MAX_DAILY_ANALYSES = 2

def check_rate_limit():
    user_id = hashlib.md5(st.context.headers.get("User-Agent", "unknown").encode()).hexdigest()
    current_time = time.time()
    
    if 'limits' not in st.session_state:
        st.session_state.limits = {}
    
    user_data = st.session_state.limits.get(user_id, {'count': 0, 'reset': current_time + 86400})
    
    if current_time > user_data['reset']:
        user_data = {'count': 0, 'reset': current_time + 86400}
    
    if user_data['count'] >= MAX_DAILY_ANALYSES:
        hours = int((user_data['reset'] - current_time) / 3600)
        return False, f"You've hit your daily limit. Thanks for using SumPro! Resets in {hours}h."
    
    user_data['count'] += 1
    st.session_state.limits[user_id] = user_data
    return True, f"{MAX_DAILY_ANALYSES - user_data['count']} analyses left today"

def init_state():
    defaults = {'messages': [], 'store': None, 'chunks': [], 'mode': None, 'summary': None, 'structure': None}
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

def extract_text(pdfs):
    text = ''
    for pdf in pdfs:
        doc = fitz.open(stream=io.BytesIO(pdf.read()), filetype="pdf")
        text += ''.join(page.get_text() or page.get_text(flags=fitz.TEXT_PRESERVE_WHITESPACE) for page in doc) + '\n'
    return text

def create_store(text):
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
    return FAISS.from_texts(chunks, OpenAIEmbeddings()), chunks

def get_context(store, queries, section=None):
    docs = []
    for q in queries:
        docs.extend(store.similarity_search(f"{section} {q}" if section else q, k=4))
    return "\n\n".join(list(dict.fromkeys([d.page_content for d in docs]))[:20])

MODES = {
    'professional': {
        'queries': ['action items decisions', 'attendees participants', 'blockers issues', 'next steps timeline', 'key outcomes results'],
        'prompt': """You are analyzing a work document. Provide a comprehensive summary that captures all important information someone would need if they missed this meeting or didn't read this document.

Start with a brief overview of what this document is about and why it exists.

Then cover what actually happened or was discussed. Be specific - use real names, actual numbers, exact dates, and concrete details. Don't just say "decisions were made" - explain what the decisions were and why they matter.

If there are action items, list each one with:
- Who is responsible (actual names)
- What they need to do (be specific)
- When it's due (actual dates or timeframes)
- Why it matters (context)

If there are open questions or blockers, explain what they are and why they're important.

End with what happens next - what are the immediate next steps, follow-up meetings, or deadlines people need to know about.

Write naturally and conversationally. This should read like a detailed briefing from a colleague who was there, not a form to fill out."""
    },
    'tech': {
        'queries': ['core concepts definitions', 'technical implementation details', 'how it works architecture', 'examples use cases', 'methods algorithms'],
        'prompt': """You are analyzing a technical document. Your goal is to help someone understand what this document is teaching, how things work, and what they can do with this knowledge.

Start by explaining what this document is about - what problem does it solve, what technology does it describe, or what concept does it teach?

Then dive into the substance. Explain the key technical concepts in detail:
- What are they and why do they matter?
- How do they actually work? (mechanisms, processes, flows)
- What are the important technical details someone needs to know?
- If there are algorithms or methods, explain what they do and when you'd use them

If the document covers implementation or architecture:
- Describe how things are structured or built
- Explain the technical decisions and tradeoffs
- Include specific technologies, frameworks, or tools mentioned

If there are examples or use cases, walk through them - they often contain the most valuable practical information.

End with practical takeaways: What can someone do with this information? When would they use this approach? What are the key things to remember?

Write like you're explaining this to a technically capable colleague. Be thorough but clear. Focus on understanding, not just listing facts."""
    },
    'digest': {
        'queries': ['main topic thesis argument', 'key points findings', 'important details facts', 'conclusions implications', 'examples evidence'],
        'prompt': """You are analyzing a document and need to capture what really matters.

Start with the main point - what is this document actually about? What's the core message or main argument?

Then explain the key points in detail. Don't just list them - explain each one:
- What is the point?
- Why does it matter?
- What evidence or examples support it?
- How does it connect to the bigger picture?

Include specific details, numbers, names, or examples that make the content concrete and memorable. These details are often what makes something worth reading.

If the document has a conclusion or recommendations, explain them clearly.

End with why this matters - what should someone take away from this? How might it affect their thinking or actions?

Write conversationally and naturally. This should read like you're telling someone about something interesting you just read, not filling out a summary template."""
    }
}

def summarize(store, mode, section=None):
    config = MODES[mode]
    context = get_context(store, config['queries'], section)
    section_note = f"\nFocus on: {section}\n" if section else ""
    prompt = f"{config['prompt']}{section_note}\n\nContext:\n{context}\n\nSummary:"
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=2000).invoke(prompt).content

def generate_widget(store, mode, summary, widget_type):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    
    if widget_type == 'email':
        ctx = "\n".join([d.page_content for d in store.similarity_search("decisions actions", k=3)])
        prompt = f"""Draft professional follow-up email:

Summary: {summary[:500]}
Context: {ctx[:800]}

Include: recap, action items with owners, next steps

Email:"""
        return llm.invoke(prompt).content
    
    elif widget_type == 'questions':
        prompts = {
            'professional': "Generate 3 clarifying questions probing decisions and assumptions",
            'tech': "Generate 3 deep technical questions about implementation and approaches",
            'digest': "Generate 3 follow-up questions about implications and applications"
        }
        prompt = f"{prompts[mode]}\n\nSummary: {summary[:400]}\n\nOne per line, no numbering:\n"
        response = llm.invoke(prompt).content
        return [q.strip().lstrip('0123456789.-) ') for q in response.split('\n') if q.strip()][:3]
    
    elif widget_type == 'concepts':
        ctx = "\n".join([d.page_content for d in store.similarity_search("key concepts", k=5)])
        prompt = f"List 3-5 key concepts with brief explanations:\n\nContext: {ctx[:1000]}\n\nFormat: Concept: Explanation\n"
        return llm.invoke(prompt).content
    
    elif widget_type == 'structure':
        sample = store.similarity_search("table of contents chapters", k=3)
        ctx = "\n\n".join([d.page_content for d in sample])
        prompt = f"Identify main sections. Return numbered list:\n1. [Title] - [Description]\n\nContext: {ctx[:1500]}\n\nMax 8:\n"
        response = llm.invoke(prompt).content
        return [line.strip() for line in response.split('\n') if line.strip() and line[0].isdigit()]

def answer_question(store, question, history):
    context = "\n\n".join([d.page_content for d in store.similarity_search(question, k=5)])
    hist = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in history[-4:]]) if len(history) > 1 else ""
    history_part = f"History:\n{hist}\n\n" if hist else ""
    prompt = f"{history_part}Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2).invoke(prompt).content

def main():
    st.set_page_config(page_title="SumPro", page_icon="📄", layout="wide")
    init_state()
    
    # Custom CSS for better visuals
    st.markdown("""
        <style>
        .main-header {
            font-size: 7rem;
            font-weight: 800;
            color: #1f1f1f;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.03em;
        }
        .creator-line {
            font-size: 1.1rem;
            color: #666;
            text-align: center;
            font-style: italic;
            margin-bottom: 3rem;
        }
        .intro-text {
            font-size: 1.05rem;
            color: #333;
            line-height: 1.8;
        }
        .feature-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
        }
        .feature-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .feature-desc {
            font-size: 0.95rem;
            line-height: 1.5;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <h1 style="font-size: 70px; font-weight: 800; text-align: center; color: #1f1f1f; margin-bottom: 3px; letter-spacing: -2px;">
            SumPro
        </h1>
        <p style="font-size: 16px; color: #666; text-align: center; font-style: italic; margin-bottom: 18px;">
            Your summarization tool created by Apinke
        </p>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Upload")
        pdfs = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
        
        st.divider()
        
        modes = {"Professional": "professional", "Tech Deep-Dive": "tech", "Quick Digest": "digest"}
        selected = st.radio("Mode", list(modes.keys()))
        mode = modes[selected]
        
        st.divider()
        
        if st.button("Analyze", type="primary", use_container_width=True):
            if not pdfs:
                st.error("Upload at least one PDF")
            else:
                allowed, msg = check_rate_limit()
                if not allowed:
                    st.error(msg)
                    return
                
                with st.spinner("Processing..."):
                    try:
                        text = extract_text(pdfs)
                        if not text.strip():
                            st.error("No text found")
                            return
                        
                        st.session_state.store, st.session_state.chunks = create_store(text)
                        st.session_state.mode = mode
                        st.session_state.summary = summarize(st.session_state.store, mode)
                        st.session_state.structure = None
                        st.session_state.messages = [{
                            "role": "assistant",
                            "content": f"**{selected} Analysis**\n\n{st.session_state.summary}",
                            "type": "summary"
                        }]
                        st.success(msg)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        if st.session_state.store:
            st.divider()
            st.metric("Chunks", len(st.session_state.chunks))
            if st.button("New Analysis", use_container_width=True):
                for key in ['messages', 'store', 'chunks', 'summary', 'structure']:
                    st.session_state[key] = [] if key == 'messages' or key == 'chunks' else None
                st.rerun()
    
    if st.session_state.store:
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                if msg.get("type") == "summary" and idx == 0:
                    st.markdown("---")
                    st.markdown("**What do you want to do with this?**")
                    
                    cols = st.columns(4 if st.session_state.mode == 'professional' else 3)
                    
                    with cols[0]:
                        if st.button("Ask me questions", key="q"):
                            questions = generate_widget(st.session_state.store, st.session_state.mode, st.session_state.summary, 'questions')
                            text = "**Here are some questions to help you think deeper:**\n\n" + "\n".join([f"- {q}" for q in questions])
                            st.session_state.messages.append({"role": "assistant", "content": text})
                            st.rerun()
                    
                    if st.session_state.mode == 'professional':
                        with cols[1]:
                            if st.button("Draft follow-up email", key="e"):
                                email = generate_widget(st.session_state.store, st.session_state.mode, st.session_state.summary, 'email')
                                st.session_state.messages.append({"role": "assistant", "content": f"**Here's a draft email you can use:**\n\n{email}"})
                                st.rerun()
                    
                    with cols[1 if st.session_state.mode != 'professional' else 2]:
                        if st.session_state.mode == 'tech' and st.button("Show key concepts", key="c"):
                            concepts = generate_widget(st.session_state.store, st.session_state.mode, st.session_state.summary, 'concepts')
                            st.session_state.messages.append({"role": "assistant", "content": f"**The key concepts you need to know:**\n\n{concepts}"})
                            st.rerun()
                    
                    with cols[2 if st.session_state.mode != 'professional' else 3]:
                        if st.session_state.mode == 'tech' and st.button("See document structure", key="s"):
                            if not st.session_state.structure:
                                st.session_state.structure = generate_widget(st.session_state.store, st.session_state.mode, st.session_state.summary, 'structure')
                            
                            if st.session_state.structure:
                                text = "**Document Structure:**\n\n" + "\n".join(st.session_state.structure)
                                text += "\n\n*Want to explore a section? Just ask - for example, 'tell me about section 2'*"
                                st.session_state.messages.append({"role": "assistant", "content": text})
                            else:
                                st.session_state.messages.append({"role": "assistant", "content": "Couldn't find a clear structure in this document. But you can still ask me specific questions about it."})
                            st.rerun()
        
        if question := st.chat_input("Ask me anything about this document..."):
            st.session_state.messages.append({"role": "user", "content": question})
            
            with st.chat_message("user"):
                st.markdown(question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    section_match = re.search(r'section (\d+)|(\d+)(?:st|nd|rd|th)?\s+section', question.lower())
                    
                    if section_match and st.session_state.structure:
                        num = int(section_match.group(1) or section_match.group(2)) - 1
                        if 0 <= num < len(st.session_state.structure):
                            section = st.session_state.structure[num]
                            answer = summarize(st.session_state.store, st.session_state.mode, section)
                            answer = f"**Here's what you need to know about that section:**\n\n{answer}"
                        else:
                            answer = "That section number doesn't exist. Check the structure I showed you earlier to see what's available."
                    else:
                        answer = answer_question(st.session_state.store, question, st.session_state.messages)
                    
                    st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
    
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="intro-text">', unsafe_allow_html=True)
            st.markdown("""
            I built this tool to solve a problem I kept running into - spending hours reading entire documents just to extract a few key insights.
    
            Simply pick your analysis mode on the left, upload your PDF, and watch it give you a summary tailored to what you need - action items from meetings, technical breakdowns, or just the key points, whichever one you want.
            
            Then use our intelligent feature which immediately suggests follow-up questions and helps you dig into more specific sections.
            
            **Get started now.**
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
                    <p style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">Professional Mode</p>
                    <p style="font-size: 0.95rem; line-height: 1.5; margin: 0;">Meeting notes? Work documents? Get action items, decisions, and who needs to do what.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
                    <p style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">Tech Deep-Dive</p>
                    <p style="font-size: 0.95rem; line-height: 1.5; margin: 0;">Technical docs? Research papers? Break down concepts, implementation details, and how it all works.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
                    <p style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">Quick Digest</p>
                    <p style="font-size: 0.95rem; line-height: 1.5; margin: 0;">Just need the highlights? Get the main points and why they matter. Fast.</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()