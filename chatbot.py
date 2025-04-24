from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv
import re

load_dotenv()

# Load dataset
df = pd.read_csv("solar_data.csv")

# OpenAI Client (OpenRouter)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI()

# Request model
class UserQuery(BaseModel):
    question: str

# Extract location using regex
def extract_location(text: str) -> str:
    for location in df["Location"].unique():
        if re.search(rf"\b{location.lower()}\b", text.lower()):
            return location
    return None

# Extract number based on keyword
def extract_number_from_question(question: str, keyword: str) -> float:
    pattern = rf"{keyword}.*?(\d+)"
    match = re.search(pattern, question.lower())
    return float(match.group(1)) if match else None

# Core info fetcher
def get_solar_info_from_question(question: str) -> str:
    location = extract_location(question)
    budget = extract_number_from_question(question, "budget")
    usage = extract_number_from_question(question, "usage")

    filtered_df = df.copy()

    if location:
        filtered_df = filtered_df[filtered_df["Location"].str.lower() == location.lower()]
    if budget:
        filtered_df = filtered_df[filtered_df["Budget_PKR"] <= budget]
    if usage:
        filtered_df = filtered_df[filtered_df["Usage_kWh_per_month"] >= usage]

    if not filtered_df.empty:
        row = filtered_df.iloc[0]
        return (
            f"In {row['Location'] if location else 'a recommended area'}, "
            f"avg sunlight: {row['Sunlight_Hours']} hours/day.\n"
            f"User's budget: PKR {budget if budget else 'N/A'}, "
            f"monthly usage: {usage if usage else 'N/A'} kWh.\n"
            f"Recommended system: {row['Recommended_System']} "
            f"(supports up to {row['Usage_kWh_per_month']} kWh/mo, cost ~PKR {row['Budget_PKR']})."
        )
    else:
        return "Sorry, I couldn't find suitable solar system data based on the provided information."

@app.post("/ask")
async def ask_solar_bot(user_query: UserQuery):
    solar_info = get_solar_info_from_question(user_query.question)

    prompt = f"""
    Given the following data from our solar dataset:
    {solar_info}

    The user asked: "{user_query.question}"

    Please respond with a clear, helpful solar system recommendation.
    """

    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"response": response.choices[0].message.content.replace("*", "")}
