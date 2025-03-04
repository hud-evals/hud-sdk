import os
import openai
from typing import Literal, Optional

ResponseType = Literal["STOP", "CONTINUE"]

class ResponseAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.async_client = openai.AsyncClient(api_key=self.api_key)
        
        self.system_prompt = """
        You are an assistant that helps determine the appropriate response to an agent's message.
        
        You will receive messages from an agent that is performing tasks for a user.
        Your job is to analyze these messages and respond with one of the following:
        
        - STOP: If the agent indicates it has successfully completed a task, even if phrased as a question
          like "I have returned to the previous website. Would you like me to do anything else?"
        
        - CONTINUE: If the agent is asking for clarification before proceeding with a task
          like "I'm about to clear cookies from this website. Would you like me to proceed?"
        
        Respond ONLY with one of these two options.
        """
    
    async def determine_response(self, agent_message: str) -> ResponseType:
        try:
            response = await self.async_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Agent message: {agent_message}\n\nWhat is the appropriate response?"}
                ],
                temperature=0.1,  # Low temperature for more deterministic responses
                max_tokens=5  # We only need a short response
            )
            
            response_text = response.choices[0].message.content.strip().upper()
            
            # Validate the response
            if "STOP" in response_text:
                return "STOP"
            else:
                return "CONTINUE"
                
        except Exception as e:
            print(f"Error determining response: {e}")
            return "CONTINUE"

