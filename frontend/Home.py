import streamlit as st
from api.general import wait_for_backend
from utils.session import init_session_state, render_sidebar

# Page Configuration
st.set_page_config(
    page_title="CarroGPT | Customer Chatbot",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded",
)


def main():
    # Wait for backend to be ready
    wait_for_backend()

    # Session State Initialization and Sidebar
    init_session_state()
    render_sidebar()

    # --------------------------------------------------------------------
    # Header
    # --------------------------------------------------------------------
    st.title("CarroGPT | Customer Chatbot")
    st.markdown(
        """
        _Questions about used cars, pricing, financing, or anything in between? No worries!_
        _CarroGPT is here to help, drawing on Carro’s knowledge base and live web search for the latest and most accurate info._
        """
    )
    st.caption(
        "Disclaimer: This is a proof of concept and NOT an official Carro product."
    )
    st.divider()

    # --------------------------------------------------------------------
    # How It Works
    # --------------------------------------------------------------------
    st.markdown("## How It Works")
    st.markdown(
        """
        CarroGPT uses a **Retrieval-Augmented Generation (RAG)** pipeline.
        We transform your documents into vector embeddings and store them in a vectorstore.
        When a query is made, we retrieve the most relevant context via semantic similarity.
        Our chatbot then uses this context, along with live web data if needed to generate an accurate response.
        """
    )
    st.divider()

    # --------------------------------------------------------------------
    # AGENT SELECTION SECTION
    # --------------------------------------------------------------------
    st.markdown("## Choose Your Fighter")

    col_react, col_crag = st.columns(2, gap="large")

    # --- Column 1: ReAct Agent ---
    with col_react:
        st.subheader("ReAct", divider="red")
        st.image(
            "assets/react_agent.png",
            caption="For fast, dynamic answers",
        )
        st.markdown(
            """
            **ReAct** stands for **Reasoning and Acting**. 
            This agent operates in a dynamic loop, thinking about the next best action and then calling the right tool (FAQ lookup, web search) to get the job done.
            """
        )
        st.markdown(
            """
            - **Pros:** Fast and highly adaptable, easy to implement with Langgraph's _create_react_agent()_.
            - **Cons:** More unpredictable path to the answer, less control.
            """
        )
        st.page_link(
            page="pages/1_ReAct_Agent.py",
            label="Try it out →",
            use_container_width=True,
        )

    # --- Column 2: Corrective RAG Agent ---
    with col_crag:
        st.subheader("Corrective RAG", divider="blue")
        st.image(
            "https://miro.medium.com/v2/resize:fit:1400/0*RYC8hbx_JksVareu.png",
            caption="For accurate, well-supported answers",
        )
        st.markdown(
            """
            **Corrective RAG** prioritizes accuracy, following a strict **Analyze → Retrieve → Grade → Generate** workflow. 
            It self-corrects (what we like to call, "ownself check ownself") by grading document relevance and using web search if needed.
            """
        )
        st.markdown(
            """
            - **Pros:** Minimizes hallucinations, Higher accuracy, and more control over the response.
            - **Cons:** Slower and more computationally expensive, harder to implement.
            """
        )
        st.page_link(
            page="pages/2_CRAG_Agent(s).py",
            label="Try it out →",
            use_container_width=True,
        )
    st.divider()

    # --------------------------------------------------------------------
    # STEP 3: Knowledge Base Management
    # --------------------------------------------------------------------
    col_img_kb, col_txt_kb = st.columns([3, 4], gap="large")
    with col_img_kb:
        st.image(
            "https://miro.medium.com/v2/resize:fit:1152/1*HLbScqT29hlYoPlOkLjPDg.png",
            width=300,
            use_container_width=False,
            caption="Organize Carro’s FAQ data",
        )
    with col_txt_kb:
        st.subheader("Manage Knowledge Base", divider="blue")
        st.write(
            "Upload, update, and manage your document collections here. "
            "Collections serves as the definitive **source of truth** that grounds the system's responses in your specific content."
        )
        st.page_link(
            page="pages/3_Knowledge_Base.py",
            label="Manage Knowledge Base →",
            icon="🗃️",
            use_container_width=True,
        )
    st.divider()

    # --------------------------------------------------------------------
    # Footer
    # --------------------------------------------------------------------
    st.caption("CarroGPT: shifting your used car buying experience into high gear!")
    st.caption(
        "Built with _Lanngraph_, _Qdrant_, and _FastAPI_ by Ren Hwa. For more details and fun projects, check out my [GitHub Repository](https://github.com/sunraymoonbeam?tab=repositories) "
    )


if __name__ == "__main__":
    main()
