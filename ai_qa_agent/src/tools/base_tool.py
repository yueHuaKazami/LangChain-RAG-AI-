"""工具基类与接口定义。

定义所有 Agent 工具应遵循的统一接口契约。
LangChain 的 StructuredTool 本身已提供标准接口，此处补充项目级约定：
工具构建函数统一以 build_<tool_name>(...) 命名，返回 StructuredTool，
并具备完善的类型提示，便于依赖注入和灵活替换。
"""
from typing import Protocol

from langchain_core.tools import BaseTool


class ToolBuilder(Protocol):
    """工具构建器协议。

    所有具体工具的构建函数应遵循此协议：接收所需依赖，返回一个 BaseTool。
    这样 Agent 调度层可以统一对待所有工具，支持灵活替换。
    """

    def __call__(self, *args, **kwargs) -> BaseTool:
        """构建并返回一个工具实例。"""
        ...
