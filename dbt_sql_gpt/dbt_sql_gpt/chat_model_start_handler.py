from langchain_core.callbacks import BaseCallbackHandler
from pyboxen import boxen


def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))
    return


class ChatModelStartHandler(BaseCallbackHandler):
    def __init__(self, memory):
        self.memory = memory

    def add_function_message_to_memory(self, function_name, result):
        if result == "no result":
            return
        function_message = f"{result}"
        self.memory.save_context({"input": function_message}, {"output": function_message})

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # print("\n\n\n\n========= Sending Messages =========\n\n")

        for message in messages[0]:
            if message.type == "system":
                boxen_print(message.content, title=message.type, color="yellow")

            elif message.type == "human":
                boxen_print(message.content, title=message.type, color="green")

            elif message.type == "ai" and "function_call" in message.additional_kwargs:
                call = message.additional_kwargs["function_call"]
                # self.add_function_message_to_memory(call['name'], call['arguments'])
                boxen_print(
                    f"Running tool {call['name']} with args {call['arguments']}",
                    title=message.type,
                    color="cyan"
                )

            elif message.type == "ai":
                boxen_print(message.content, title=message.type, color="blue")

            elif message.type == "function":
                self.add_function_message_to_memory(message.type, message.content)
                boxen_print(message.content, title=message.type, color="purple")

            else:
                boxen_print(message.content, title=message.type)
