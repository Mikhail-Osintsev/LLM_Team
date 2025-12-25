from typing import Dict, Any, List, Callable
from app.backend.core.retriever import retrieve


class MCPServer:
    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],  
        func: Callable,
    ) -> None:
        self.tools[name] = {
            "description": description,
            "parameters": parameters,  
            "func": func,
        }

    def list_tools(self) -> Dict[str, Any]:
        return {
            "tools": [
                {
                    "name": name,
                    "description": t["description"],
                    "parameters": t["parameters"],  
                }
                for name, t in self.tools.items()
            ]
        }

    def run_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        func = self.tools[name]["func"]
        return func(**args)


class MCPClient:
    def __init__(self, server: MCPServer) -> None:
        self.server = server

    def list_tools(self) -> List[Dict[str, Any]]:
        resp = self.server.list_tools()
        return resp["tools"]

    def run_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.server.run_tool(name, args)



def retrieve_tool(query: str = "", top_k: int = 4) -> Dict[str, Any]:
    # Иногда LLM вызывает tool без обязательных аргументов — не падаем.
    if not isinstance(query, str) or not query.strip():
        return {"passages": []}

    passages = retrieve(query, top_k=top_k)
    return {"passages": passages}


mcp_server = MCPServer()

mcp_server.register_tool(
    name="retrieve",
    description="Ищет релевантные фрагменты текста книги по запросу.",
    parameters={    
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Текстовый вопрос или запрос для поиска по книге.",
            },
            "top_k": {
                "type": "integer",
                "description": "Сколько фрагментов вернуть.",
                "minimum": 1,
                "maximum": 10,
                "default": 4,
            },
        },
        "required": ["query"],
    },
    func=retrieve_tool,
)

mcp_client = MCPClient(mcp_server)



def list_tools_for_llm() -> List[Dict[str, Any]]:
    """
    Удобная обёртка: вернуть список тулов так, как его ожидает планировщик.
    В будущем здесь можно заменить реализацию на настоящий MCP-клиент.
    """
    return mcp_client.list_tools()


def call_tool_from_llm(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Удобная обёртка: вызвать инструмент по имени с аргументами.
    Тоже легко заменить на реальный протокол в будущем.
    """
    return mcp_client.run_tool(name, args) 