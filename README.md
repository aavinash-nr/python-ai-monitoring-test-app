# python-ai-chat-lambda

This project contains two main Lambda functions that demonstrate different approaches to interacting with Amazon Bedrock AI models:

## Core Files

- **app.py** - Uses direct Amazon Bedrock runtime client to invoke Claude models
- **ai.py** - Uses LangChain framework with ChatBedrock for model conversations

## Monitoring

This project is instrumented with New Relic for performance monitoring and observability.

## Quick Start

To deploy this application:

```bash
sam build --use-container
sam deploy --guided
```

## Local Testing

Test locally with:
```bash
sam local start-api
sam local invoke HelloWorldFunction --event events/event.json
```

## Testing

Run tests with:
```bash
pip install -r tests/requirements.txt --user
python -m pytest tests/unit -v
```

## Cleanup

To delete the deployed application:
```bash
sam delete --stack-name "python-ai-chat-lambda"
```
