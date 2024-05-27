# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import subprocess


from langchain_core.runnables.base import RunnableSequence
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate, \
    SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

class AI:
    def __init__(self, human_prompt, memory_path):

        # chat = ChatOpenAI(model="gpt-4")

        chat = Ollama(model="llama3:70b")

        memory = ConversationBufferMemory(
            chat_memory=FileChatMessageHistory(memory_path),
            memory_key="messages",
            return_messages=True
        )
        prompt = ChatPromptTemplate(
            input_variables=["content", "messages"],
            messages=[
                MessagesPlaceholder(variable_name="messages"),
                SystemMessagePromptTemplate.from_template(
                    "The conversation will be about Kubernetes. Please ignore other unrelated questions."
                    "\n Every question I ask make sure that you have the concrete Kubernetes Context and namespace information, if not, ask me for those instead of answering my prompt."
                    "\n Every kubernetes command you give me, use the namespace and context I provided already."
                    "\n When you give me a script to run, don't give me markdown, or even word response, just the code itself."
                    "\n You will prefix all code answers with '> '"),
                HumanMessagePromptTemplate.from_template(
                    human_prompt)
            ]
        )

        self.chain = LLMChain(
            llm=chat,
            prompt=prompt,
            memory=memory
        )

        # chain = prompt | chat


class MyShell:
    def __init__(self):
        self.session = PromptSession(history=InMemoryHistory())
        self.commands = {
            'exit': self.exit_shell,
        }
        self.command_completer = WordCompleter(list(self.commands.keys()), ignore_case=True)
        self.chain_for_commands = AI("Generate a shell script that {content}", "messages.json").chain
        self.chain_more_generic = AI("{content}", "messages.json").chain

    def execute_command(self, command):
        print(f"Executing ... {command}")
        response = subprocess.run(command.split(), capture_output=True, text=True)
        print(response.stdout)
        print(response.stderr)
    def exit_shell(self, arg):
        print('Thank you for using my shell.')
        return True

    def default(self, line, previous_response) -> str:
        if line.startswith("kubectl"):
            self.execute_command(line)
            return previous_response
        if (line.startswith(".") or ">" not in previous_response) and not line.startswith("$"):
            result = self.chain_more_generic.invoke({"content": line})
            print("message: ")
        else:
            result = self.chain_for_commands.invoke({"content": line})
            print("command: ")

        print(result["text"])

        if result['text'].startswith("> "):
            self.execute_command(result['text'].replace("> ", ""))
        elif result['text'].startswith(">"):
            self.execute_command(result['text'].replace(">", ""))
        return result["text"]

    def run(self):
        previous_response = ""
        while True:
            try:
                user_input = self.session.prompt('k8s >> ', auto_suggest=AutoSuggestFromHistory(), completer=self.command_completer)
                if user_input:
                    command, *args = user_input.split()
                    if command in self.commands:
                        if self.commands[command](" ".join(args)):
                            break
                    else:
                        previous_response = self.default(user_input, previous_response)
            except (EOFError, KeyboardInterrupt):
                print('Exiting shell.')
                break


if __name__ == '__main__':
    shell = MyShell()
    shell.run()

