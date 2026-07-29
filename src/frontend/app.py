import asyncio
import streamlit as st

from src.adapters.listing_generator import LLMHomeListingGenerator
from src.core.entities.models import BuyerPreferences
from src.infrastructure.container import AppContainer
from src.infrastructure.config import get_settings
from src.frontend.matching import top_listing_matches


QUESTIONS = [
    ("house_size", "How big do you want your house to be?"),
    ("priorities", "What are the 3 most important things for you in choosing this property?"),
    ("amenities", "Which amenities would you like?"),
    ("transportation", "Which transportation options are important to you?"),
    ("neighborhood_type", "How urban do you want your neighborhood to be?"),
]


def _init_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "recommendation" not in st.session_state:
        st.session_state.recommendation = ""


def _build_preferences() -> BuyerPreferences:
    answers = st.session_state.answers
    return BuyerPreferences(
        house_size=answers["house_size"],
        priorities=answers["priorities"],
        amenities=answers["amenities"],
        transportation=answers["transportation"],
        neighborhood_type=answers["neighborhood_type"],
    )


async def _run_recommendation(preferences: BuyerPreferences) -> str:
    container = AppContainer.create_demo(preferences=preferences)
    await container.ensure_home_listings()
    result = await container.engine.recommend(preferences)
    return result.answer


async def _regenerate_listings() -> None:
    generator = LLMHomeListingGenerator(settings=get_settings())
    await generator.generate_home_listings()


def _question_step(step: int) -> None:
    key, prompt = QUESTIONS[step]
    st.subheader(f"Step {step + 1} of {len(QUESTIONS)}")
    answer = st.text_area(
        prompt,
        value=st.session_state.answers.get(key, ""),
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", disabled=step == 0, use_container_width=True):
            st.session_state.step = max(0, step - 1)
            st.rerun()
    with col2:
        if st.button("Next", use_container_width=True):
            if not answer.strip():
                st.warning("Please provide an answer to continue.")
                return
            st.session_state.answers[key] = answer.strip()
            st.session_state.step = min(len(QUESTIONS), step + 1)
            st.rerun()


def _review_step() -> None:
    st.subheader("Review preferences")
    for field, prompt in QUESTIONS:
        st.markdown(f"**{prompt}**")
        st.write(st.session_state.answers.get(field, ""))
        st.divider()

    if st.button("Generate Recommendation", type="primary", use_container_width=True):
        preferences = _build_preferences()
        with st.spinner("Finding your best matching home..."):
            st.session_state.recommendation = asyncio.run(_run_recommendation(preferences))
        st.success("Recommendation generated.")

    if st.session_state.recommendation:
        preferences = _build_preferences()
        matches = top_listing_matches(get_settings().data_path, preferences.to_query(), limit=3)

        st.subheader("Top 3 Listing Matches")
        if not matches:
            st.info("No ranked listing matches available yet.")
        else:
            cols = st.columns(3)
            for idx, match in enumerate(matches):
                with cols[idx]:
                    score_pct = int(match.score * 100)
                    st.markdown(
                        f"""
                        <div style="border:1px solid #d0d7de;border-radius:12px;padding:14px;">
                        <h4 style="margin-top:0;">{match.neighborhood}</h4>
                        <p><strong>Location:</strong> {match.location}</p>
                        <p><strong>Bedrooms:</strong> {match.bedrooms} | <strong>Bathrooms:</strong> {match.bathrooms}</p>
                        <p><strong>Size:</strong> {match.house_size_sqft} sqft</p>
                        <p><strong>Price:</strong> ${match.price_k_usd:.0f}k</p>
                        <p><span style="background:#e8f0fe;padding:4px 8px;border-radius:999px;">
                        Match Score: {score_pct}%</span></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.subheader("Personalized Recommendation")
        st.write(st.session_state.recommendation)


def main() -> None:
    st.set_page_config(page_title="HomeMatch", page_icon="🏡", layout="wide")
    st.title("HomeMatch - Personalized Real Estate Assistant")
    st.caption("Recruiter demo UI built with Streamlit")
    _init_state()
    with st.sidebar:
        st.header("Data Controls")
        if st.button("Regenerate Listings", use_container_width=True):
            with st.spinner("Regenerating listings with LLM..."):
                asyncio.run(_regenerate_listings())
            st.success("Listings regenerated.")

    if st.session_state.step < len(QUESTIONS):
        _question_step(st.session_state.step)
    else:
        _review_step()


if __name__ == "__main__":
    main()

