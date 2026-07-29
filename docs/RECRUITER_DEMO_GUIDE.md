# Recruiter Demo Guide

## 2-Minute Walkthrough

1. Run the app:
   - `streamlit run src/frontend/app.py`
2. Answer the 5 preference questions (multi-step flow).
3. Click **Generate Recommendation** to show:
   - personalized recommendation paragraph
   - top-3 matched properties with score badges
4. Click **Regenerate Listings** in the sidebar to show live dataset refresh.

## Talking Points

- Started from Udacity Generative AI Nanodegree project outline.
- Refactored to modular architecture with interfaces and adapters.
- Added fail-fast config validation and prompt-input sanitization.
- Implemented dependency injection and async recommendation flow.
- Built a Streamlit UI to improve product usability and presentation quality.

## What I Learned

- How to convert a notebook-style prototype into layered production code.
- How to make LLM apps safer and easier to maintain through typed contracts and validation.
- How to improve hiring signal by pairing backend quality with an interactive frontend.

## Next Improvements

- Add persistent user sessions and recommendation history.
- Improve ranking with embedding-based similarity scoring rather than token overlap.
- Add CI workflow to run tests on every pull request.

