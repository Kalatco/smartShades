# Smart Shades Agent

A LangGraph-based intelligent agent for smart shades control and automation using Azure OpenAI and Hubitat Z-Wave integration.

## Overview

This project implements an AI agent using LangGraph that can intelligently control Z-Wave smart shades through Hubitat using natural language commands. The agent integrates with Hubitat's Maker API for IoT device control and uses Azure OpenAI for intelligent command interpretation with solar intelligence features.

## Features

- **Natural Language Control**: "open the front shade", "close all blinds", "block the sun"
- **Multi-Blind Operations**: "open the side window halfway, and front window fully"
- **Room-specific and house-wide control**
- **Solar Intelligence**: Automatic sun exposure detection and glare management
- **Specific blind targeting by name or keyword**
- **RESTful API** for external integrations (Apple Shortcuts, Home Assistant, etc.)
- **Hubitat Z-Wave Integration** via Maker API
- **Voice-friendly responses** for Apple Shortcuts and Siri

## Hardware Requirements

### Hubitat Hub Setup
1. **Hubitat Elevation Hub** (C-7 or newer recommended)
2. **Z-Wave Smart Shades/Blinds** (tested with devices supporting position control)
3. **Network connectivity** between your computer and Hubitat hub

### Supported Z-Wave Shade Devices
- Any Z-Wave shade controller that supports position commands (0-100%)
- Examples: Somfy, Lutron, Hunter Douglas, etc.
- Device must be paired with your Hubitat hub

## Setup Instructions

### 1. Hubitat Configuration

#### Enable Maker API
1. Log into your Hubitat hub web interface
2. Go to **Apps** → **Add Built-In App**
3. Select **Maker API**
4. Choose your shade devices to include in the API
5. Note the **Access Token** and **API URL** (you'll need these)

#### Find Device IDs
1. In Hubitat, go to **Devices**
2. Click on each shade device
3. Note the **Device Network ID** (this is your device ID)
4. Test device control using the device page

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Hubitat Configuration
HUBITAT_ACCESS_TOKEN=your_maker_api_access_token
HUBITAT_API_URL=http://your-hubitat-ip

# API Configuration (optional)
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### 3. Blinds Configuration

Edit `blinds_config.json` to match your setup:

```json
{
  "makerApiId": "1",
  "latitude": 47.6062,
  "longitude": -122.3321,
  "timezone": "America/Los_Angeles",
  "house_orientation": "North-facing front door",
  "notes": "Guest bedroom faces east, master bedroom faces west",
  "rooms": {
    "guest_bedroom": {
      "blinds": [
        {
          "id": "123",
          "name": "Guest Side Window",
          "orientation": "North"
        },
        {
          "id": "124", 
          "name": "Guest Front Window",
          "orientation": "East"
        }
      ]
    },
    "master_bedroom": {
      "blinds": [
        {
          "id": "125",
          "name": "Master West Window", 
          "orientation": "West"
        }
      ]
    }
  }
}
```

**Configuration Notes:**
- `id`: Use the Device Network ID from Hubitat
- `name`: Friendly name for voice commands
- `orientation`: Cardinal direction for solar intelligence (North, South, East, West)
- `latitude/longitude`: Your location for sun calculations
- `timezone`: Your local timezone for accurate solar times

### 4. Installation and Running

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the agent:**
   ```bash
   python src/main.py
   ```

4. **Access the API:**
   - Swagger UI: http://localhost:8000/docs
   - API Base: http://localhost:8000

## Voice Commands Examples

- **Basic Control**: "Open the blinds", "Close all shades", "Set to 75 percent"
- **Multi-Blind**: "Open the side window halfway, and front window fully"
- **Solar-Aware**: "Block the sun", "Reduce glare", "Open but avoid direct sunlight"
- **Room-Specific**: "Close all blinds in the bedroom", "Open living room shades"

## Integration Examples

### Apple Shortcuts
1. Create new shortcut
2. Add "Get Contents of URL" action
3. Set URL: `http://your-ip:8000/rooms/guest_bedroom/control`
4. Set Method: POST
5. Set Request Body: `{"command": "Ask Siri for input"}`
6. Add "Speak Text" action with response message

### Home Assistant
```yaml
rest_command:
  smart_shades:
    url: "http://your-ip:8000/rooms/{{ room }}/control"
    method: POST
    headers:
      Content-Type: application/json
    payload: '{"command": "{{ command }}"}'
```

## Docker Deployment

Build and run with Docker:

```bash
docker build -t smart-shades-agent .
docker run -p 8000:8000 --env-file .env smart-shades-agent
```

## Project Structure

```
├── src/
│   ├── agent/          # LangGraph agent implementation
│   ├── models/         # Data models and Pydantic schemas
│   └── main.py         # FastAPI application entry point
├── blinds_config.json  # Hubitat device configuration
├── .env               # Environment variables (create from template)
├── .env.example       # Environment variables template
├── Dockerfile         # Container configuration
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Advanced Features

### Solar Intelligence
- Automatic sun position tracking
- Window-specific sun exposure analysis
- Commands like "block the sun" automatically target sunny windows
- Glare reduction based on sun intensity

### Multi-Operation Commands
- Handle complex requests: "open side window halfway, front window fully"
- Intelligent parsing of different positions for different blinds
- Scope detection (room vs house-wide commands)

## License

MIT License
