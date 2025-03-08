# Use Python 3.9
FROM python:3.9-slim

# Create a working directory for your app
WORKDIR /app

# Copy in requirements.txt first (for dependency caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose FastAPI port (if you plan to access the app externally)
EXPOSE 8000

# Start both FastAPI (via uvicorn) and the Discord bot in one container
# The ampersand (&) runs uvicorn in the background, then starts the bot.
CMD uvicorn main:app --host 0.0.0.0 --port 8000 & python discord_bot.py