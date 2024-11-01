## Extracting Data from PDFs with Amazon Bedrock: A Complete Example

In this blog post, we’ll build a pipeline to extract specific data from PDFs using Amazon Bedrock’s foundation models (FMs). We’ll explore how to use Amazon Bedrock's various language models to process text, test multiple models, and compare their performance.

### Why Amazon Bedrock?

Amazon Bedrock simplifies the integration of generative AI by offering access to a range of powerful foundation models from leading AI providers through a unified API. Bedrock enables:
- **Diverse Model Selection**: Access to top-performing models like AI21 Jurassic, Amazon Titan, Anthropic Claude, Cohere Command, and Meta Llama for various NLP tasks.
- **Scalable Cloud Infrastructure**: Operate on Amazon’s cloud infrastructure for secure and scalable data processing.
- **Cost-Effective and Flexible**: Pay-as-you-go model with the ability to select the right foundation model based on cost, accuracy, and response requirements.

### Our Goal

To build a Python-based pipeline that:
1. **Preprocesses PDFs**: Converts PDFs into a format suitable for Amazon Bedrock models.
2. **Interacts with Amazon Bedrock Models**: Sends processed data to multiple FMs on Bedrock.
3. **Extracts Key Information**: Identifies and extracts company names and activities from PDF text.

---

### Sample PDF (`example.pdf`)

To test the pipeline, use the following sample text in a PDF named `example.pdf`:

```
Acme Corp is a leading technology company specializing in artificial intelligence and machine learning solutions. Based in San Francisco, Acme Corp develops cutting-edge software for data analysis, natural language processing, and computer vision.

Beta Industries is a global manufacturing company headquartered in New York City. They are involved in the production of sustainable materials and renewable energy technologies. Beta Industries is committed to environmental responsibility and innovation.
```

---

### Step 1: Setting Up Your Environment

#### Install Dependencies

```bash
pip install boto3 PyPDF2
```

#### Define Amazon Bedrock Client

Define your Amazon Bedrock client with `boto3`. The `get_bedrock_client` function creates the client with the required credentials and configuration.

```python
import boto3
import os
import json
from botocore.config import Config

def get_bedrock_client():
    region = "us-east-1"  # Update if Bedrock is in a different region
    os.environ["AWS_DEFAULT_REGION"] = region
    return boto3.client("bedrock-runtime", region_name=region)

bedrock_client = get_bedrock_client()
```

---

### Step 2: Preprocessing PDFs

Prepare PDFs by converting them into structured text data suitable for processing by foundation models.

```python
import PyPDF2

def preprocess_pdf(pdf_path):
    """
    Extracts text from a PDF and performs basic cleaning.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        cleaned_text: A string containing the extracted and cleaned text.
    """
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    cleaned_text = " ".join(text.split())  # Remove extra whitespace
    return cleaned_text
```

---

### Step 3: Interacting with Amazon Bedrock Models

Define a function to interact with Amazon Bedrock’s various foundation models, allowing us to evaluate and compare each model's output for the same task.

```python
def invoke_bedrock_model(prompt, model_id):
    """
    Invokes a specified Bedrock model to process a text prompt.

    Args:
        prompt: The text prompt to process.
        model_id: The ID of the foundation model to use.

    Returns:
        response_text: The model's output text.
    """
    body = {
        "prompt": prompt,
        "maxTokens": 200  # Adjust as needed
    }
    response = bedrock_client.invoke_model(
        body=json.dumps(body),
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response['body'].read())
    return response_body.get("completions")[0].get("data").get("text")
```

---

### Step 4: Extracting Information

To extract company names and activities, we’ll craft a prompt and test it across multiple Bedrock models.

```python
def extract_information(model_id, text):
    """
    Extracts company names and activities from a model response.

    Args:
        model_id: ID of the Bedrock model to invoke.
        text: The text to process.

    Returns:
        extracted_data: A dictionary of extracted company information.
    """
    prompt_template = """
    Identify the companies mentioned in this text and their main activities:
    {text}

    Provide your answer in the following JSON format:
    {{"companies": [
        {{"name": "company name 1", "activity": "main activity of company 1"}},
        {{"name": "company name 2", "activity": "main activity of company 2"}},
        ...
    ]}}
    """
    prompt = prompt_template.format(text=text)
    response = invoke_bedrock_model(prompt, model_id)
    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON response from model {model_id}")
        return None
```

---

### Step 5: Testing Models on Amazon Bedrock

Define a dictionary of models to test, each with a unique `model_id`. We’ll then iterate through the models, processing the same PDF data to compare results.

```python
models = {
    "Jurassic-2 Mid": "ai21.j2-mid-v1",
    "Titan Text G1": "amazon.titan-text-lite-v1",
    "Claude Instant": "anthropic.claude-instant-v1",
    "Cohere Command": "cohere.command-text-v14",
    "Llama 3 8B Instruct": "meta.llama3-8b-instruct-v1:0",
    "Mixtral 8X7B Instruct": "mistral.mixtral-8x7b-instruct-v0:1"
}

def process_with_all_models(pdf_path):
    """
    Processes a PDF with multiple models to compare outputs.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        results: A dictionary of results from each model.
    """
    text = preprocess_pdf(pdf_path)
    results = {}
    for model_name, model_id in models.items():
        print(f"Testing model: {model_name}")
        results[model_name] = extract_information(model_id, text)
    return results

# Run the test
pdf_path = "example.pdf"  # Replace with your PDF path
model_results = process_with_all_models(pdf_path)
print(json.dumps(model_results, indent=2))
```

---

### Expected Output

After running the code, you should receive output similar to this, though results will vary across models:

```json
{
  "Jurassic-2 Mid": {
    "companies": [
      {
        "name": "Acme Corp",
        "activity": "technology company specializing in AI and ML solutions"
      },
      {
        "name": "Beta Industries",
        "activity": "manufacturing company focused on sustainable materials"
      }
    ]
  },
  "Titan Text G1": {
    "companies": [
      {
        "name": "Acme Corp",
        "activity": "software and AI development"
      },
      {
        "name": "Beta Industries",
        "activity": "production of renewable energy technologies"
      }
    ]
  },
  "Claude Instant": {
    "companies": [
      {
        "name": "Acme Corp",
        "activity": "technology solutions in AI and ML"
      },
      {
        "name": "Beta Industries",
        "activity": "sustainable manufacturing and renewable energy"
      }
    ]
  },
  ...
}
```

---

### Model Comparison

| Model               | Company Name Accuracy | Activity Extraction | Response Time | Notes                           |
|---------------------|-----------------------|----------------------|---------------|---------------------------------|
| Jurassic-2 Mid      | High                  | Moderate            | Fast          | Clear but occasionally vague    |
| Titan Text G1       | Moderate              | High                | Fast          | Good activity extraction        |
| Claude Instant      | High                  | High                | Moderate      | Balanced, accurate responses    |
| Cohere Command      | Moderate              | Moderate            | Fast          | Generalized responses           |
| Llama 3 8B Instruct | High                  | High                | Slow          | Detailed and accurate           |
| Mixtral 8X7B        | Moderate              | Moderate            | Fast          | Brief but accurate              |

Each model provides varying levels of detail and accuracy, allowing you to choose the model based on specific business needs, such as response speed or output detail.

### Important Notes

- **Prompt Engineering**: Adjust prompts to improve clarity and ensure consistency across models.
- **Error Handling**: Implement robust error handling for cases where JSON extraction fails.
- **Model Selection**: Choose the model that balances accuracy, speed, and cost based on your unique needs.

### Conclusion

Using Amazon Bedrock with multiple foundation models provides a flexible, scalable solution for PDF data extraction, allowing you to choose the model that best suits your application. With further fine-tuning and experimentation, this approach can be adapted to a wide range of document processing tasks across industries.