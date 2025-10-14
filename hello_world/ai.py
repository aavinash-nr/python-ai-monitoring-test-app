import json
import os
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

# This initialization is correct and does not need to be changed
chat_model = ChatBedrockConverse(
    region_name=os.environ.get('AWS_REGION', 'us-east-1'),
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0.5,
    max_tokens=2048,
)

def lambda_handler(event, context):
    """
    AWS Lambda handler that invokes a Bedrock model using the correct
    ChatBedrockConverse class from LangChain.
    """
    print(f"Received event: {json.dumps(event)}")

    prompt_text = 'Hello, How are you?'

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
        message = HumanMessage(content=prompt_text)
        response = chat_model.invoke([message])
        final_completion_text = response.content
        
        print(f"Successfully received response via ChatBedrockConverse. Final text: {final_completion_text}")

    except Exception as e:
        print(f"Error invoking model with ChatBedrockConverse: {str(e)}")
        return {
            'statusCode': 502,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": f"Failed to invoke model with ChatBedrockConverse: {str(e)}"})
        }

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({
            "completion": final_completion_text,
            # --- THIS IS THE CORRECTED LINE ---
            "model_id": chat_model.model_id,
            "original_prompt_received": prompt_text,
            "message": "Successfully processed request via LangChain ChatBedrockConverse."
        })
    }