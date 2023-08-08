import csv
from openai import OpenAI
import os
# Initialize the OpenAI API with your API key
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Open the CSV file
with open("kc_house_data.csv", "r") as f:
    reader = csv.reader(f)
    data = list(reader)

# Generate output based on the prompts given by the user
for row in data:
    output = openai.generate(prompt=
    
    row[0])
    print(output)