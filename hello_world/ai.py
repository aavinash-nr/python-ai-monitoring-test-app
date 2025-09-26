import json
import os
import boto3

# For better performance, initialize the client outside the handler function.
# This allows the connection to be reused across Lambda invocations.
bedrock_runtime = boto3.client(
    'bedrock-runtime',
    region_name=os.environ.get('AWS_REGION', 'us-east-1')
)

# Model configuration
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MAX_TOKENS = 2048
TEMPERATURE = 0.5

def lambda_handler(event, context):
    """
    AWS Lambda handler that invokes a Bedrock model using the direct Converse API.
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
        # Use direct Bedrock Converse API
        response = bedrock_runtime.converse(
            modelId=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt_text}]
                }
            ],
            inferenceConfig={
                "maxTokens": MAX_TOKENS,
                "temperature": TEMPERATURE
            }
        )
        
        # Extract the response content
        final_completion_text = response['output']['message']['content'][0]['text']
        
        print(f"Successfully received Bedrock Converse API response. Final text: {final_completion_text}")

    except Exception as e:
        print(f"Error invoking model with Bedrock Converse API: {str(e)}")
        return {
            'statusCode': 502,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": f"Failed to invoke model with Bedrock Converse API: {str(e)}"})
        }

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({
            "completion": final_completion_text,
            "model_id": MODEL_ID,
            "original_prompt_received": prompt_text,
            "message": "Successfully processed request via Bedrock Converse API."
        })
    }