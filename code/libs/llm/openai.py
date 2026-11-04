from openai import OpenAI


def prompt_gpt(api_key, instructions, input, model="gpt-4.1-nano", temperature=0):
    """One call to the OpenAI Responses API; returns the raw response object."""
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input,
        temperature=temperature,
    )

    return response
