import streamlit as st

from app.llm.openrouter import OpenRouterSummarizer
from app.manager import ContextManager, ContextStrategy
from app.models import MODEL_REGISTRY
from app.strategies.summarization import SummarizationStrategy


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Context Length Manager",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Premium white theme
# -------------------------------------------------------------------

st.markdown(
    """
    <style>

    /* =============================================================
       Global
    ============================================================= */

    .stApp {
        background: #f8fafc;
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* =============================================================
       Sidebar
    ============================================================= */

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #eaecf0;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 2rem 1.35rem;
    }

    .sidebar-brand {
        font-size: 21px;
        font-weight: 700;
        color: #101828;
        letter-spacing: -0.4px;
    }

    .sidebar-description {
        font-size: 12px;
        color: #667085;
        margin-top: 4px;
        margin-bottom: 24px;
    }

    .sidebar-heading {
        font-size: 11px;
        font-weight: 700;
        color: #667085;
        letter-spacing: 0.7px;
        text-transform: uppercase;
        margin-top: 22px;
        margin-bottom: 9px;
    }

    .sidebar-value {
        font-size: 14px;
        font-weight: 650;
        color: #101828;
    }

    .sidebar-label {
        font-size: 11px;
        color: #667085;
    }

    /* =============================================================
       Hero
    ============================================================= */

    .hero-label-native {
        font-size: 11px;
        font-weight: 700;
        color: #667085;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .hero-title-native {
        font-size: 40px;
        font-weight: 750;
        color: #101828;
        letter-spacing: -1.5px;
        line-height: 1.15;
        margin: 0 0 10px 0;
    }

    .hero-description-native {
        font-size: 15px;
        color: #667085;
        line-height: 1.65;
        max-width: 850px;
        margin-bottom: 22px;
    }

    /* =============================================================
       Section headings
    ============================================================= */

    .section-title-native {
        font-size: 19px;
        font-weight: 700;
        color: #101828;
        margin-top: 4px;
        margin-bottom: 4px;
    }

    .section-description-native {
        font-size: 13px;
        color: #667085;
        margin-bottom: 18px;
    }

    /* =============================================================
       Metric cards
    ============================================================= */

    .metric-card-native {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 105px;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.025);
    }

    .metric-label-native {
        font-size: 10px;
        font-weight: 700;
        color: #667085;
        text-transform: uppercase;
        letter-spacing: 0.65px;
        margin-bottom: 8px;
    }

    .metric-value-native {
        font-size: 25px;
        font-weight: 750;
        color: #101828;
        line-height: 1.2;
    }

    .metric-helper-native {
        font-size: 11px;
        color: #98a2b3;
        margin-top: 6px;
    }

    /* =============================================================
       Strategy cards
    ============================================================= */

    .strategy-description-native {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 12px;
        padding: 14px 17px;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .strategy-description-title {
        font-size: 13px;
        font-weight: 700;
        color: #101828;
        margin-bottom: 4px;
    }

    .strategy-description-text {
        font-size: 12px;
        color: #667085;
        line-height: 1.55;
    }

    /* =============================================================
       Buttons
    ============================================================= */

    .stButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 650;
        font-size: 14px;
    }

    .stButton > button[kind="primary"] {
        background: #101828;
        border-color: #101828;
        color: #ffffff;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1d2939;
        border-color: #1d2939;
    }

    /* =============================================================
       Inputs
    ============================================================= */

    textarea {
        border-radius: 12px !important;
    }

    [data-baseweb="select"] > div {
        border-radius: 10px;
    }

    /* =============================================================
       Progress
    ============================================================= */

    [data-testid="stProgressBar"] {
        margin-top: 8px;
        margin-bottom: 12px;
    }

    /* =============================================================
       Expander
    ============================================================= */

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 12px;
    }

    /* =============================================================
       Footer
    ============================================================= */

    .footer-native {
        text-align: center;
        color: #98a2b3;
        font-size: 11px;
        padding-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def format_number(value: int) -> str:
    """Format an integer with thousands separators."""

    return f"{value:,}"


def create_manager(model_name: str) -> ContextManager:
    """
    Create a context manager.

    OpenRouter summarization is initialized only when the
    summarization strategy is selected.
    """

    return ContextManager.from_model(
        model_name=model_name,
    )


def create_summarization_manager(
    model_name: str,
) -> ContextManager:
    """Create a context manager with OpenRouter summarization."""

    summarizer = SummarizationStrategy(
        summarizer=OpenRouterSummarizer()
    )

    return ContextManager.from_model(
        model_name=model_name,
        summarization_strategy=summarizer,
    )


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">Context Manager</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-description">'
        'LLM Context Optimization'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-heading">Model</div>',
        unsafe_allow_html=True,
    )

    model_name = st.selectbox(
        "Model",
        options=list(MODEL_REGISTRY.keys()),
        label_visibility="collapsed",
    )

    model_config = MODEL_REGISTRY[model_name]

    st.markdown(
        '<div class="sidebar-heading">Configuration</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="sidebar-label">Context Window</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sidebar-value">'
            f'{format_number(model_config.context_window)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="sidebar-label">Input Budget</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sidebar-value">'
            f'{format_number(model_config.max_input_tokens)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="sidebar-label">Output Budget</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sidebar-value">'
            f'{format_number(model_config.max_output_tokens)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="sidebar-label">Safety Margin</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sidebar-value">'
            f'{format_number(model_config.safety_margin)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-heading">System</div>',
        unsafe_allow_html=True,
    )

    st.caption("Token-aware processing")
    st.caption("Truncation + Chunking + Summarization")
    st.caption("OpenRouter integration")


# -------------------------------------------------------------------
# Hero
# -------------------------------------------------------------------

st.markdown(
    '<div class="hero-label-native">'
    'LLM CONTEXT OPTIMIZATION'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="hero-title-native">'
    'Context Length Manager'
    '</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-description-native">'
    'Analyze token usage and intelligently manage oversized '
    'LLM context using truncation, chunking, or AI-powered '
    'summarization.'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# -------------------------------------------------------------------
# Model Overview
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title-native">Model Overview</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description-native">'
    'Current model configuration and available context capacity.'
    '</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-card-native">
            <div class="metric-label-native">
                Model
            </div>
            <div class="metric-value-native"
                 style="font-size:18px;">
                {model_config.model_name}
            </div>
            <div class="metric-helper-native">
                Selected LLM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card-native">
            <div class="metric-label-native">
                Context Window
            </div>
            <div class="metric-value-native">
                {format_number(model_config.context_window)}
            </div>
            <div class="metric-helper-native">
                Total capacity
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card-native">
            <div class="metric-label-native">
                Input Budget
            </div>
            <div class="metric-value-native">
                {format_number(model_config.max_input_tokens)}
            </div>
            <div class="metric-helper-native">
                Usable input
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card-native">
            <div class="metric-label-native">
                Output Budget
            </div>
            <div class="metric-value-native">
                {format_number(model_config.max_output_tokens)}
            </div>
            <div class="metric-helper-native">
                Reserved output
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# -------------------------------------------------------------------
# Context Input
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title-native">Context Input</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description-native">'
    'Paste a document, conversation, source code, logs, or any '
    'other text that will be processed by the model.'
    '</div>',
    unsafe_allow_html=True,
)

text = st.text_area(
    "Context",
    height=280,
    placeholder=(
        "Paste your document, conversation, source code, "
        "logs, or other context here..."
    ),
    label_visibility="collapsed",
)


# -------------------------------------------------------------------
# Token Analysis
# -------------------------------------------------------------------

if text.strip():

    manager = create_manager(model_name)

    metrics = manager.process(text).metrics

    if metrics is not None:

        st.markdown(
            '<div class="section-title-native">Token Analysis</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description-native">'
            'Current input compared with the available model budget.'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Input Tokens",
                format_number(metrics.input_tokens),
            )

        with col2:
            st.metric(
                "Available",
                format_number(metrics.max_input_tokens),
            )

        with col3:
            st.metric(
                "Remaining",
                format_number(metrics.remaining_tokens),
            )

        with col4:
            st.metric(
                "Utilization",
                f"{metrics.utilization_percent:.1f}%",
            )

        utilization = min(
            metrics.utilization_percent / 100,
            1.0,
        )

        st.progress(utilization)

        if metrics.fits:

            st.success(
                "Context is within the available input budget."
            )

        else:

            st.warning(
                "Context exceeds the available input budget by "
                f"{format_number(metrics.overflow_tokens)} tokens."
            )


st.divider()


# -------------------------------------------------------------------
# Optimization Strategy
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title-native">'
    'Optimization Strategy'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description-native">'
    'Choose how oversized context should be handled.'
    '</div>',
    unsafe_allow_html=True,
)

strategy = st.radio(
    "Strategy",
    options=[
        ContextStrategy.TRUNCATE,
        ContextStrategy.CHUNK,
        ContextStrategy.SUMMARIZE,
    ],
    format_func=lambda value: value.value.title(),
    horizontal=True,
    label_visibility="collapsed",
)


if strategy == ContextStrategy.TRUNCATE:

    st.markdown(
        """
        <div class="strategy-description-native">
            <div class="strategy-description-title">
                Truncation
            </div>
            <div class="strategy-description-text">
                Removes tokens beyond the available input budget
                while preserving the beginning of the context.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chunk_size = None
    overlap = 0


elif strategy == ContextStrategy.CHUNK:

    st.markdown(
        """
        <div class="strategy-description-native">
            <div class="strategy-description-title">
                Chunking
            </div>
            <div class="strategy-description-text">
                Splits large context into smaller token-based
                segments with optional overlap.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        chunk_size = st.number_input(
            "Chunk size",
            min_value=1,
            max_value=model_config.max_input_tokens,
            value=min(
                1000,
                model_config.max_input_tokens,
            ),
            step=100,
        )

    with col2:

        overlap = st.number_input(
            "Overlap",
            min_value=0,
            max_value=max(0, chunk_size - 1),
            value=min(
                100,
                max(0, chunk_size - 1),
            ),
            step=10,
        )


else:

    st.markdown(
        """
        <div class="strategy-description-native">
            <div class="strategy-description-title">
                AI Summarization
            </div>
            <div class="strategy-description-text">
                Uses the configured OpenRouter model to compress
                the context while preserving important information.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chunk_size = None
    overlap = 0


st.divider()


# -------------------------------------------------------------------
# Process Context
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title-native">Process Context</div>',
    unsafe_allow_html=True,
)

if st.button(
    "Optimize Context",
    type="primary",
    use_container_width=True,
):

    if not text.strip():

        st.error(
            "Please enter some context before processing."
        )

        st.stop()

    if strategy == ContextStrategy.CHUNK:

        if overlap >= chunk_size:

            st.error(
                "Overlap must be smaller than chunk size."
            )

            st.stop()

    with st.spinner("Optimizing context..."):

        try:

            if strategy == ContextStrategy.SUMMARIZE:

                manager = create_summarization_manager(
                    model_name
                )

            else:

                manager = create_manager(
                    model_name
                )

            result = manager.process(
                text=text,
                strategy=strategy,
                chunk_size=(
                    chunk_size
                    if strategy == ContextStrategy.CHUNK
                    else None
                ),
                overlap=(
                    overlap
                    if strategy == ContextStrategy.CHUNK
                    else 0
                ),
            )

        except Exception as exc:

            st.error(
                f"Processing failed: {exc}"
            )

            st.stop()


    # ---------------------------------------------------------------
    # Optimization Result
    # ---------------------------------------------------------------

    st.markdown(
        '<div class="section-title-native">'
        'Optimization Result'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description-native">'
        'Comparison of the original context and processed output.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Original Tokens",
            format_number(
                result.original_token_count
            ),
        )

    with col2:

        st.metric(
            "Final Tokens",
            format_number(
                result.final_token_count
            ),
        )

    with col3:

        reduction = (
            (
                result.original_token_count
                - result.final_token_count
            )
            / result.original_token_count
            * 100
            if result.original_token_count
            else 0
        )

        st.metric(
            "Token Reduction",
            f"{reduction:.1f}%",
        )


    # ---------------------------------------------------------------
    # Chunk result
    # ---------------------------------------------------------------

    if result.strategy == ContextStrategy.CHUNK.value:

        st.success(
            f"{len(result.chunks or [])} chunks generated."
        )

        for chunk in result.chunks or []:

            with st.expander(
                f"Chunk {chunk.index + 1} "
                f"• {chunk.token_count} tokens"
            ):

                st.write(chunk.text)


    # ---------------------------------------------------------------
    # Text result
    # ---------------------------------------------------------------

    else:

        st.text_area(
            "Processed Context",
            value=result.text,
            height=320,
            label_visibility="collapsed",
        )

        if result.strategy == ContextStrategy.SUMMARIZE.value:

            st.info(
                "The context was compressed using the configured "
                "OpenRouter model."
            )

        elif result.strategy == ContextStrategy.TRUNCATE.value:

            st.info(
                "The context was truncated to fit the available "
                "input-token budget."
            )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.markdown(
    '<div class="footer-native">'
    'Context Length Manager · '
    'Token-aware LLM context optimization'
    '</div>',
    unsafe_allow_html=True,
)