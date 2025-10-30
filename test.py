'''
import asyncio
from agents import Agent, Runner

CODE_IMPLEMENTER_MODEL = "gpt-4o"

async def main():
    # Step 1: Create the agent
    coding_agent = Agent(
        name="ResearchAssistant",
        instructions="You are a helpful research assistant that listens well.",
        model=CODE_IMPLEMENTER_MODEL,
    )

    # Step 2: Define messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Print Hi",
                },
            ],
        },
    ]

    # Step 3: Run the agent
    result = await Runner.run(starting_agent=coding_agent, input=messages)

    print(result.raw_responses[0].usage)
    

# Step 4: Execute the async function
if __name__ == "__main__":
    asyncio.run(main())
'''

import asyncio
from agents import Agent, Runner

# === Globals for tracking ===
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens_used = 0
token_cost = 0.0

# Example cost per 1K tokens (adjust for your model)
input_cost = 0.005 / 1000
output_cost = 0.015 / 1000

CODE_IMPLEMENTER_MODEL = "gpt-4o"

async def run_agent_test():
    global total_prompt_tokens, total_completion_tokens, total_tokens_used, token_cost

    # Create agent
    coding_agent = Agent(
        name="ResearchAssistant",
        instructions="You are a helpful research assistant that listens well.",
        model=CODE_IMPLEMENTER_MODEL,
    )

    # Create input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Print Hi"},
            ],
        },
    ]

    # Run the agent
    result = await Runner.run(starting_agent=coding_agent, input=messages)

    # Track tokens
    tokens_used = 0
    if hasattr(result, "raw_responses") and result.raw_responses:
        resp = result.raw_responses[0]
        if hasattr(resp, "usage") and resp.usage:
            usage = resp.usage
            tokens_used = usage.total_tokens
            total_prompt_tokens += usage.input_tokens
            total_completion_tokens += usage.output_tokens
            total_tokens_used += usage.total_tokens
            token_cost += (usage.input_tokens * input_cost) + (usage.output_tokens * output_cost)

    print(f"✅ Agent ran successfully w/ {tokens_used} tokens")
    print(f"Total Tokens: {total_tokens_used} | Estimated Cost: ${token_cost:.6f}")

    return True

if __name__ == "__main__":
    asyncio.run(run_agent_test())
