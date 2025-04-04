# import discord
# import requests
# import os
# import json
# # Discord bot token
# # DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
# # Read tokens from file
# config_path = "config.json"
# if os.path.exists(config_path):
#     with open(config_path, "r") as f:
#         config = json.load(f)
#     DISCORD_TOKEN = config.get("DISCORD_TOKEN", "")
# else:
#     DISCORD_TOKEN = ""
#
#
# # FastAPI endpoint
# FASTAPI_URL = "http://127.0.0.1:8000/interact"
#
# # Initialize Discord client
# intents = discord.Intents.default()
# intents.messages = True
# client = discord.Client(intents=intents)
#
#
# @client.event
# async def on_ready():
#     print(f"{client.user} is now connected!")
#
#
# @client.event
# async def on_message(message):
#     # Ignore bot's own messages
#     if message.author == client.user:
#         return
#
#     # Check if the message starts with "Socrates"
#     if len(message.content) > 1: #message.content.lower().startswith("socrates"):
#         user_input = message.content[len("socrates "):].strip()
#
#         # Send user input to FastAPI
#         response = requests.post(FASTAPI_URL, json={"user_input": user_input})
#
#         if response.status_code == 200:
#             data = response.json()
#             await message.channel.send(f"Socrates: {data['response']}")
#         else:
#             await message.channel.send("Socrates encountered an error. Please try again later.")
#
#
# # Run Discord bot
# client.run(DISCORD_TOKEN)



import discord
import requests
import os
import json

# Load configuration from config.json
config_path = "config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    DISCORD_TOKEN = config.get("DISCORD_TOKEN", "")
else:
    DISCORD_TOKEN = ""

# URL for the FastAPI endpoint
FASTAPI_URL = "http://127.0.0.1:8000/interact"

intents = discord.Intents.default()
intents.messages = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"{client.user} is now connected to Discord!")

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    user_id = str(message.author.id)
    user_input = message.content.strip()

    payload = {
        "user_id": user_id,
        "user_input": user_input
    }
    try:
        response = requests.post(FASTAPI_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            await message.channel.send(f"Socrates: {data['response']}")
        else:
            await message.channel.send("Socrates encountered an error. Please try again.")
    except Exception as e:
        await message.channel.send(f"Error contacting Socrates API: {str(e)}")

client.run(DISCORD_TOKEN)