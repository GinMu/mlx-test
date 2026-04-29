import argparse
from datetime import datetime

from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache

# Parse command line arguments
parser = argparse.ArgumentParser(description="MLX LM Tool Calling Demo")

parser.add_argument("--question", nargs="?", help="The question to ask the model")
parser.add_argument("--model", default="Qwen3.5-4B-MLX-8bit", help="Path or name of the model to load")
parser.add_argument("--output", default="output.txt", help="Output file path")
args = parser.parse_args()

output_file = open(args.output, "a")

def log(*args):
    text = " ".join(str(a) for a in args)
    output_file.write(text + "\n")

# Load the model and tokenizer
model_path = args.model
model, tokenizer = load(model_path)


def current_date():
    """
    A function that returns the current date and time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_wallet():
    """
    A function that generates a new wallet address.
    """
    return "0x1234567890abcdef1234567890abcdef12345678"


tools = {"current_date": current_date, "generate_wallet": generate_wallet}

# User prompt
messages = [{"role": "user", "content": args.question}]

log("\n====== New Conversation ======\n")
log("model:", model_path)
log("user question:", args.question)

# Step 1: Generate prompt with tools
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tools=list(tools.values()),
)

prompt_cache = make_prompt_cache(model)

# Step 2: Model generates tool call
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=False,
    prompt_cache=prompt_cache,
)

log("")
log("model output:")
log(response)

# Step 3: Parse and execute tool call
start_tool = response.find(tokenizer.tool_call_start) + len(tokenizer.tool_call_start)
end_tool = response.find(tokenizer.tool_call_end)

if start_tool == -1 or end_tool == -1:
    exit(1)

tool_call = tokenizer.tool_parser(response[start_tool:end_tool].strip())
tool_result = tools[tool_call["name"]](**tool_call["arguments"])
log(f"\ntool calling: {tool_call['name']}() -> {tool_result}")

# Step 4: Append tool result to conversation
messages.append({"role": "assistant", "content": response})
messages.append({"role": "tool", "name": tool_call["name"], "content": tool_result})
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

log("\ngenerate result:")
# Step 5: Model generates final response
final_response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=512,
    verbose=True,
    prompt_cache=prompt_cache,
)
log(final_response)

log("\n====== End of Conversation ======\n")

output_file.close()
