from openai import OpenAI

def prompt_gpt(api_key, instructions, input, model="gpt-4.1-nano", temperature=0):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=api_key,
    )

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input,
        temperature=temperature,
    )

    return response