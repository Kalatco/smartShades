# Smart Shades Agent

A LangGraph-based intelligent agent for smart shades control and automation using Azure OpenAI.

## Overview

This project implements an AI agent using LangGraph that can intelligently control smart shades based on natural language commands. The agent integrates with Hubitat for IoT device control and uses Azure OpenAI for intelligent command interpretation.

## Features

- Natural language shade control ("open the front shade", "close all blinds")
- Room-specific and house-wide control
- Specific blind targeting by name or keyword
- RESTful API for external integrations
- Hubitat IoT integration
- Structured output using Pydantic models

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Azure OpenAI and Hubitat configuration
   ```

3. Configure your blinds in `blinds_config.json`

4. Run the agent:
   ```bash
   python src/main.py
   ```

## Configuration

### Azure OpenAI
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Your deployment name
- `AZURE_OPENAI_API_VERSION`: API version (e.g., 2024-02-15-preview)

### Hubitat Integration
Configure your blinds in `blinds_config.json` with room mappings and device IDs.

## Docker Deployment

Build and run with Docker:

```bash
docker build -t smart-shades-agent .
docker run -p 8000:8000 smart-shades-agent
```

## Project Structure

```
├── src/
│   ├── agent/          # LangGraph agent implementation
│   ├── models/         # Data models
│   └── main.py         # Application entry point
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variables template
```

## License

MIT License
