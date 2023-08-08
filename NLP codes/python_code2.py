
import json

# Create the JSON data
data = {
  "document": {
    "title": "Sample PDF",
    "author": "John Doe",
    "pages": [
      {
        "number": 1,
        "content": "This is the content of page 1"
      },
      {
        "number": 2,
        "content": "This is the content of page 2"
      }
    ]
  }
}

# Save the JSON data into a file
with open(r'C:\NLP codes\document.json', 'w') as file:
    json.dump(data, file)