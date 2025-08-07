import openai
from openai import OpenAI
import time
import random

client = OpenAI()


def generate_response_multiagent(engine, max_tokens, system_role, user_input):
    print("Generating response for engine: ", engine)
    start_time = time.time()
    response = client.chat.completions.create(
                    model=engine,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    messages=[  
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_input}
                    ],
                    timeout = 200
                )
    end_time = time.time()
    print('Finish!')
    print("Time taken: ", end_time - start_time)

    return response

class api_handler:
    def __init__(self, model):
        self.model = model

        if self.model == 'instructgpt':
            self.engine = 'text-davinci-002'
        elif self.model == 'instructgpt-gen':
            self.engine = 'text-davinci-002'
        elif self.model == 'newinstructgpt':
            self.engine = 'text-davinci-003'
        elif self.model == 'oldinstructgpt':
            self.engine = 'text-davinci-001'
        elif self.model == 'gpt3':
            self.engine = 'davinci'
        elif self.model == 'codex':
            self.engine = 'code-davinci-002'
        elif self.model == 'gpt3-edit':
            self.engine = 'text-davinci-edit-001'
        elif self.model == 'codex-edit':
            self.engine = 'code-davinci-edit-001'
        elif self.model == 'chatgpt':
            self.engine = 'gpt-35-turbo-16k'
        elif self.model == 'gpt4.1':
            self.engine = 'gpt-4.1'
        else:
            raise NotImplementedError

    def get_output_multiagent(self, system_role, user_input, max_tokens):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = generate_response_multiagent(self.engine, max_tokens, system_role, user_input)
                if response.choices and response.choices[0].message and response.choices[0].message.content != "":
                    return response.choices[0].message.content
                else:
                    return "ERROR." 
            except (openai.APITimeoutError, openai.APIConnectionError, openai.APIError, Exception) as error:
                print(f'Attempt {attempt+1} of {max_attempts} failed with error: {error}')
                if attempt == max_attempts - 1:
                    return "ERROR."