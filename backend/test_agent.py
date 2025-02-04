import os
import sys

from dotenv import load_dotenv
from langchain.tools import Tool

load_dotenv()

from langchain_openai import ChatOpenAI
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_community.utilities import GoogleSerperAPIWrapper

# Create agent
def initialize_agent(wallet_data: str = None):

    print("Initializing agent... with wallet data: ", wallet_data)
    """Initialize the agent with CDP Agentkit.
    
    Args:
        wallet_data (str, optional): The CDP wallet data to initialize the agent with.
            If None, a new wallet will be created.
    """
    # Initialize LLM.
    llm = ChatOpenAI(model=os.getenv("MODEL_NAME"), api_key=os.getenv("OPENAI_API_KEY"))  

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    all_tools = cdp_toolkit.get_tools()
    
    # Filter to only use specific CDP tools (customize this list as needed)
    wanted_tool_names = [
        "get_balance",
        "transfer",
        "get_wallet_details",
    ]
    
    tools = [tool for tool in all_tools if tool.name in wanted_tool_names]

    # Add Google Serper API tool
    search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
    search_tool = Tool(
        name="Search",
        description="Search the internet for current information. Use this tool when you need to find information about current events, facts, or anything that requires up-to-date information.",
        func=search.run
    )
    tools.append(search_tool)

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=(
            "You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit. "
            "You are empowered to interact onchain using your tools. If you ever need funds, you can request "
            "them from the faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet "
            "details and request funds from the user. Before executing your first action, get the wallet details "
            "to see what network you're on. If there is a 5XX (internal) HTTP error code, ask the user to try "
            "again later. If someone asks you to do something you can't do with your currently available tools, "
            "you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, "
            "recommend they go to docs.cdp.coinbase.com for more information. Be concise and helpful with your "
            "responses. Refrain from restating your tools' descriptions unless it is explicitly requested."
        ),
    ), config


# Chat Mode
def run_chat_mode(agent_executor, config):
    """Run the agent interactively based on user input."""
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nPrompt: ")
            if user_input.lower() == "exit":
                break

            # Run agent with the user's input in chat mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)

def main():
    """Start the chatbot agent."""
    agent_executor, config = initialize_agent()  # Creates a new wallet by default
    run_chat_mode(agent_executor=agent_executor, config=config)

# Example of how to use with a specific wallet:
def run_with_wallet(wallet_data: str):
    """Run the agent with a specific wallet.
    
    Args:
        wallet_data (str): The CDP wallet data to use.
    """
    agent_executor, config = initialize_agent(wallet_data=wallet_data)
    run_chat_mode(agent_executor=agent_executor, config=config)

if __name__ == "__main__":
    # Load wallet data from file
    try:
        with open("wallet_data.txt", "r") as f:
            wallet_data = f.read().strip()
    except FileNotFoundError:
        print("No wallet data file found. Please create a wallet first.")
        sys.exit(1)
    run_with_wallet(wallet_data=wallet_data)