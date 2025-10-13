import json
import os
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

# For better performance, initialize the client outside the handler function.
# This allows the connection to be reused across Lambda invocations.
chat_model = ChatBedrock(
    # Use the AWS region from the environment variable, default to us-east-1
    region_name=os.environ.get('AWS_REGION', 'us-east-1'),
    # Specify the model ID
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    # Pass model-specific parameters using model_kwargs
    model_kwargs={
        "max_tokens": 2048,
        "temperature": 0.5,
    },
)

def lambda_handler(event, context):
    """
    AWS Lambda handler that invokes a Bedrock model using the LangChain framework.
    """
    print(f"Received event: {json.dumps(event)}")

    # Default prompt text
    prompt_text = 'Hello, How are you?'

    # Robustly extract the prompt from the event
    if isinstance(event, dict) and 'prompt' in event:
        prompt_text = event.get('prompt', prompt_text)
    elif 'body' in event and event['body'] is not None:
        try:
            body_data = json.loads(event['body'])
            if 'prompt' in body_data:
                prompt_text = body_data['prompt']
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing event['body']: {str(e)}. Using default prompt.")

    print(f"Final prompt being used: \"{prompt_text}\"")
    
    try:
        # --- CHANGE: Use LangChain to invoke the model ---
        # 1. Create a LangChain message from the prompt
        message = HumanMessage(content=prompt_text)
        
        # 2. Invoke the model with the message
        response = chat_model.invoke([message])
        
        # 3. The response content is directly available on the object
        final_completion_text = response.content
        
        print(f"Successfully received LangChain response. Final text: {final_completion_text}")

    except Exception as e:
        print(f"Error invoking model with LangChain: {str(e)}")
        return {
            'statusCode': 502,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": f"Failed to invoke model with LangChain: {str(e)}"})
        }

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({
            "completion": final_completion_text,
            "model_id": chat_model.model_id,
            "original_prompt_received": prompt_text,
            "message": "Successfully processed request via LangChain."
        })
    }