from dotenv import load_dotenv
from openai import OpenAI
import os   

load_dotenv("api_key.env")  # Load the environment variables from the .env file

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY')) # Initialize the OpenAI client

# Format it within a list of dictionaries to be passed to the API
# Here we know what the user is going to say, so we can add it right into the messages list
# Also for simple prompts like this, we don't need to add a system message to the messages list
messages = [
    {
        "role": "system", 
        "content": "You are a lyricist assistant/rap ghoswriter. Your job is take the user input and turn into rhyming poetry with an avantgarde witty, clever, comedic, gritty, and dark tone."
    
        "role": "user", 
        "content": "Spit some lyrics."
    }
]

# Call the API - make sure we are using the gpt-3.5-turbo model
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

# Print the response message
print(completion.choices[0].message.content)