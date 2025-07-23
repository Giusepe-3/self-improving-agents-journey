from openai import OpenAI

client = OpenAI(api_key="")

response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "web_search_preview"}],
    input="What is the tempature in Valencia, Spain?",
)

print(response.output_text)