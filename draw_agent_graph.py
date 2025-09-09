"""
Script to draw the Smart Shades Agent V2 LangGraph
"""

import sys
import os
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent.smart_shades_agent_v2 import SmartShadesAgentV2


async def draw_agent_graph():
    """Draw the Smart Shades Agent V2 graph"""
    print("Initializing Smart Shades Agent V2...")

    try:
        # Create agent instance
        agent = SmartShadesAgentV2()

        # Initialize the agent (this builds the graph)
        await agent.initialize()

        print("Agent initialized successfully!")
        print("Drawing graph...")

        # Draw the graph to a PNG file
        image_data = agent.graph.get_graph().draw_mermaid_png()

        output_file = "smart_shades_agent_v2_graph.png"
        with open(output_file, mode="wb") as f:
            f.write(image_data)

        print(f"Graph saved to: {output_file}")

        # Also save as mermaid text for reference
        mermaid_text = agent.graph.get_graph().draw_mermaid()
        mermaid_file = "smart_shades_agent_v2_graph.mmd"
        with open(mermaid_file, mode="w") as f:
            f.write(mermaid_text)

        print(f"Mermaid text saved to: {mermaid_file}")

        # Print the mermaid text to console as well
        print("\nMermaid Graph Definition:")
        print("=" * 50)
        print(mermaid_text)
        print("=" * 50)

        # Shutdown the agent
        await agent.shutdown()

    except Exception as e:
        print(f"Error drawing graph: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(draw_agent_graph())
