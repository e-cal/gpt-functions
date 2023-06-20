# %% [markdown]
"""
# Setup
"""

# %%
import json
import os

import openai

try:
    with open(os.path.expanduser("~/.cache/oai"), "r") as f:
        openai.api_key = f.read().strip()
except:
    print("Error reading openai api key from ~/.cache/oai")
    exit(1)

# %% [markdown]
"""
# The Functions API
"""

# %% [markdown]
r"""
## The intended use case

As presented by OpenAI, functions in the chat models are meant to be used to get
the model to generate function arguments.

The idea is you ask the model a question and provide it function descriptions,
then if the question seems suitable for a function, the model will generate
the arguments for the function rather than answer the question directly.

The dev can then easily pass the generated arguments to the function and call it,
either to provide the output to the user, or to pass back to the model and get 
another response with the result in context.

\
**TLDR**
- meant to be used to get the model to generate function arguments
- ask the model a question and provide it function descriptions
- if the question seems suitable for a function, the model will generate args
- then pass args to the function
- give result to user, or pass back to model and get another response
"""

# %% [markdown]
"""
To illustrate, here a simplified version of the example OpenAI gives in their docs:
"""


# %%
# dummy function to demonstrate
def get_weather(location, unit="c"):
    """Get the weather in a location"""
    # in reality you'd call some api here with the args
    weather = {
        "location": location,
        "unit": unit,
        "temperature": 20,
        "condition": "sunny",
    }
    return weather


# %%
msg = "What's the weather like in Kingston?"
res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613", # until old models are deprec, use -0613
    messages=[{"role": "user", "content": msg}],
    functions=[  # list of dicts
        {
            "name": "get_weather",  # name of function
            "description": "Get the weather in a location",  # description of function
            "parameters": {  # parameters of function
                "type": "object",  # type of parameters (almost always object)
                "properties": {  # the function arguments
                    "location": {  # arg
                        "type": "string",  # arg type (any json type)
                        "description": "The location to get the weather for",  # arg description
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to return the temperature in",
                        # can enumerate the possible values instead of leaving openended
                        "enum": ["f", "c"],
                    },
                },
                "required": ["location"],  # list required args
            },
        }
    ],
    function_call="auto",  # "auto" or a function name
)

res_content = res.choices[0].message.to_dict()  # type: ignore
res_content

# %% [markdown]
"""
when it calls a function, there is nothing in the content field, only function_call

(and vice versa)

so, if we want to then call the function and show the user:
"""

# %%
func_to_call = res_content["function_call"]["name"]
args = json.loads(res_content["function_call"]["arguments"])

weather = globals()[func_to_call](**args)
weather

# %%
# and so you could show it back to the user like this:
print(
    f"The weather in {args['location']} is {weather['temperature']}°{weather['unit']} and {weather['condition']}"
)

# %% [markdown]
"""
This alone is pretty cool and very useful. Like a more powerful/extensible version of
plugins in chatGPT that we can write ourselves for any usecase. 

Some stuff you guys might want to use it for is using wolfram alpha, wikipedia,
investopedia, or looking up cases or medical information in a database etc.

Sort of just substitutes langchain in a lot of places.
"""

# %% [markdown]
"""
## The more powerful use case

The new versions of the models which have the functions api were of course finetuned 
on tons of data to give these structured argument outputs. So, we can exploit that
to get the model to generate any structured data we want. I don't think I can overstate
how useful this is when trying to get it to conform to some output form, for pretty much
any application.

_The caveat is we need to "trick" it into thinking it's generating function arguments._
"""

# %% [markdown]
"""
Here's how I used it in my work
"""

# %%
sys_prompt = """
Before answering, you should think through the question step-by-step.
Explain your reasoning at each step towards answering the question.
If calculation is required, do each step of the calculation as a step in your reasoning.
Finally, indicate the correct answer
"""

question = "let a = b+c. if a = c^2 + 4 and c = b^3 - 7 what is b?"

res = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ],
    functions=[
        {
            "name": "answer_question",
            # tell it what the "function" should do
            "description": "Thinks through and answers a multiple choice question on finance",
            "parameters": {
                "type": "object",
                "properties": {  # use args to tell it what fields it should generate
                    "thinking": {
                        "type": "array",  # arrays will be arbitrary length
                        "items": {
                            "type": "string",
                            "description": "Thought and/or calculation for a step in the process of answering the question",
                        },
                        "description": "Step by step thought process and calculations towards answering the question",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The answer to the question",
                        # use enum to restrain its output for easy parsing
#                         "enum": ["A", "B", "C"],
                    },
                },
                "required": ["thinking", "answer"],
            },
        }
    ],
    function_call={"name": "answer_question"},
)
ans = res.choices[0].message.to_dict()["function_call"]["arguments"]  # type: ignore
out = json.loads(ans)
out

# %%
for i, line in enumerate(out["thinking"]):
    print(f"{i+1}. {line}")

print("\nAnswer:", out["answer"])

# %%
answer = "B"
if out["answer"] == answer:
    print("Correct")
else:
    print("Failed")

# %% [markdown]
"""
### Without functions
"""

# %%
# sys_prompt last line: "Finally output the correct answer"
sys_prompt_nofunc = (
    sys_prompt + " in brackets as such: [[answer]] where answer is A, B, or C"
)

res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": sys_prompt_nofunc},
        {"role": "user", "content": question},
    ],
)
print(res.choices[0].message.content)

# %% [markdown]
"""
While _most_ of the time it will listen to your formatting request, it's not guaranteed
and often adds some verbose explanation around your formatted output. 

This makes parsing annoyingly inconsistent, and for applications more complicated than this
it may introduce countless edge cases and issues. Overall, its just not a headache 
you need to deal with anymore.
"""

# %% [markdown]
"""
Even for user queries it can be nicer to use functions to get clean structured output
instead of messy inconsistent text. How about getting code help?
"""

# %%
question = "In python, calculate the square root of 144"

# %%
sys_prompt = """You are a helpful programming assistant that can answer questions about code.
When the question is about syntax or how to do something in a specific language, you should
respond only with the code. Otherwise give an answer as normal.
"""
res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ],
)
print(res.choices[0].message.content)

# %%
res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        # {"role": "system", "content": sys_prompt}, # don't even need a sys prompt
        {"role": "user", "content": question},
    ],
    functions=[
        {
            "name": "code_help",
            "description": "Gives the corresponding code for an answer to a question about programming",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Lines of code constituting the answer to the coding question",
                    },
                },
                "required": ["code"],
            },
        }
    ],
    function_call={"name": "code_help"},
)
ans = res.choices[0].message.to_dict()["function_call"]["arguments"]  # type: ignore
out = json.loads(ans)
print(out["code"])
