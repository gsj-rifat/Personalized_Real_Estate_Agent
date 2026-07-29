import asyncio
import streamlit as st

from src.core.entities.models import BuyerPreferences
from src.infrastructure.container import AppContainer


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
        st.subheader("Personalized Recommendation")
        st.write(st.session_state.recommendation)


def main() -> None:
    st.set_page_config(page_title="HomeMatch", page_icon="🏡", layout="wide")
    st.title("HomeMatch - Personalized Real Estate Assistant")
    st.caption("Recruiter demo UI built with Streamlit")
    _init_state()

    if st.session_state.step < len(QUESTIONS):
        _question_step(st.session_state.step)
    else:
        _review_step()


if __name__ == "__main__":
    main()

