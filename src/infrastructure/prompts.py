BUYER_QUESTIONS = [
    "How big do you want your house to be?",
    "What are the 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",
]

RECOMMENDATION_TEMPLATE = """\
You are a professional real estate sales assistant helping a home buyer.
Use the retrieved listings and the customer's stated preferences to suggest \
the single best matching home.
Be concise, warm, and persuasive. Maximum 5 sentences.

Retrieved Listings:
{context}

Customer Preferences:
{chat_history}

Question: {question}

Answer:\
"""

LISTING_GENERATION_TEMPLATE = """\
Generate a CSV table about {topic} with these columns: {attributes}.
Use realistic US real estate data. Generate exactly {rows} rows.
Output only the raw CSV content — no markdown, no explanation.\
"""
