from phi.agent import Agent
from phi.model.cohere import CohereChat
from phi.tools.duckduckgo import DuckDuckGo

agent = Agent(
    model=CohereChat(id="command-r-08-2024",api_key='cohere_api_key'),
    tools=[DuckDuckGo()],
    markdown=True,
)

agent.print_response(
    "https://media.istockphoto.com/id/664994770/photo/hes-got-crime-on-his-mind.jpg?s=2048x2048&w=is&k=20&c=Y-M17fmExvmmWAvjC7eblB1pijE8UrDBOKmHeylFiCE=",
    images=["test.jpg"],
    stream=True,
)

from phi.agent import Agent
from phi.model.cohere import CohereChat
from phi.tools.duckduckgo import DuckDuckGo

agent = Agent(
model=CohereChat(id="command-r-08-2024",api_key='cohere_api_key'),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
)

agent.print_response("Tell me something about neuromorphic chips",stream=True)