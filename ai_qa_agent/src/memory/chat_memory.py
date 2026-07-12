"""对话记忆管理。

维护多轮对话的完整消息历史，保证模型能理解上下文追问
（如"它还有什么缺点？"中的"它"指代上一轮的主题）。

提供统一的记忆接口，供 RAG Agent 和 RAG Chain 共用。
"""
from langchain_core.messages import BaseMessage


class ChatMemory:
    """短期对话记忆，保存完整消息历史。

    每轮把新的用户输入或模型回复追加进去，一起传给模型，
    使模型能看到完整的上下文链条。
    """

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []

    def add(self, message: BaseMessage) -> None:
        """追加一条消息到历史。

        Args:
            message: 要追加的消息（HumanMessage / AIMessage 等）。
        """
        self._messages.append(message)

    def add_many(self, messages: list[BaseMessage]) -> None:
        """批量追加消息。

        Args:
            messages: 要追加的消息列表。
        """
        self._messages.extend(messages)

    @property
    def messages(self) -> list[BaseMessage]:
        """返回当前完整对话历史（只读视图）。"""
        return list(self._messages)

    def replace(self, messages: list[BaseMessage]) -> None:
        """用新消息列表替换当前历史。

        用于 Agent 模式：agent.invoke 返回包含完整工具交互过程的消息列表，
        用它替换历史以保证下一轮能看到本轮的工具调用记录。

        Args:
            messages: 新的消息列表。
        """
        self._messages = list(messages)

    def clear(self) -> None:
        """清空对话历史。"""
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)
