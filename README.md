# 7008project
# **GPT-2 ESG Chatbot - README**

## **Overview**

This repository contains a Python-based implementation of fine-tuning a GPT-2 model for answering questions related to ESG (Environmental, Social, and Governance). It includes training the GPT-2 model on a custom text dataset, generating responses to user prompts, and deploying a chatbot interface using Gradio.

---

## **Features**

1. **Fine-tune GPT-2**: The model is fine-tuned on a custom dataset (`training_data.txt`) for ESG-related topics.
2. **Text Generation**: The model can generate coherent text responses to user queries based on the fine-tuned knowledge.
3. **Chatbot Interface**: A user-friendly chatbot powered by Gradio, where users can interact with the model by asking questions about ESG.
4. **Customizable Parameters**: Easily adjust training, generation, and chatbot settings.

---

## **File Structure**

```
.
├── training_data.txt            # Dataset for fine-tuning GPT-2
├── train.py                     # Script for fine-tuning the GPT-2 model
├── chatbot.py                   # Script for deploying the Gradio chatbot
├── gpt2-esg/                    # Directory for saving the fine-tuned GPT-2 model
├── logs/                        # Directory for logging during training
├── README.md                    # This README file
...
```

---

## **Setup and Installation**

### **Prerequisites**

- Python 3.8 or higher
- `pip` (Python package manager)
- GPU (optional but recommended for training)

### **Installation**

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-repo/gpt2-esg-chatbot.git
   cd gpt2-esg-chatbot
   ```

2. **Install required packages**:
   ```bash
   pip install torch transformers gradio
   ```

3. **Prepare the training dataset**:
   - Create a text file named `training_data.txt` and add ESG-related text data (questions and answers, articles, or any relevant content).
   - Ensure the dataset is in a plain-text format.

---

## **Training the GPT-2 Model**

The `train.py` script handles the fine-tuning process for GPT-2.

### **Steps to Train:**

1. Open the `train.py` file and ensure the paths and parameters are configured correctly:
   - Default dataset: `training_data.txt`
   - Model output directory: `./gpt2-esg`
   
2. Run the training script:
   ```bash
   python train.py
   ```

3. **Output**:
   - The fine-tuned model and tokenizer will be saved in the `./gpt2-esg` directory.
   - Training logs will be saved in the `./logs` directory.

---

## **Using the Chatbot**

The chatbot is implemented using Gradio and can be launched with the `chatbot.py` script.

### **Steps to Launch the Chatbot:**

1. Ensure the fine-tuned model is saved in the `./gpt2-esg` directory.
2. Run the chatbot script:
   ```bash
   python chatbot.py
   ```

3. Open the Gradio interface in your browser. You can either:
   - Access the local link provided in the terminal, e.g., `http://127.0.0.1:7860`
   - Use the public Gradio shareable link if enabled.

4. **Usage**:
   - Type your question in the chatbox (e.g., "How can I improve my company's environmental performance?").
   - The chatbot will generate a response based on the fine-tuned model.

---

## **Customizing Parameters**

### **Training Parameters** (in `train.py`):
- `num_train_epochs`: Number of epochs for training (default: 1).
- `per_device_train_batch_size`: Batch size for training (default: 2).
- `learning_rate`: Learning rate for the optimizer (default: 1e-4).

### **Text Generation Parameters** (in `generate_text` or `generate_response`):
- `max_length`: Maximum length of the generated text (default: 200).
- `temperature`: Sampling temperature; lower values make the output more deterministic (default: 0.7).
- `no_repeat_ngram_size`: Prevents repetition of n-grams (default: 2).

### **Chatbot Interface**:
- You can customize the Gradio chatbot's title, description, and examples in `chatbot.py`.

---

## **Examples**

### **Training Example**

To train the model with a custom dataset named `esg_data.txt`:
1. Replace the dataset file path in `train.py`:
   ```python
   train_dataset = load_dataset('esg_data.txt', tokenizer)
   ```
2. Run the training script:
   ```bash
   python train.py
   ```

### **Chatbot Example**

Ask the chatbot a question like:
- **Input**: "What are the key aspects of corporate governance?"
- **Output**: "Corporate governance involves accountability, transparency, fairness, and responsibility in managing a company."

---

## **Dependencies**

The following Python libraries are used in this project:

- [Transformers](https://huggingface.co/docs/transformers): For working with the GPT-2 model.
- [Torch](https://pytorch.org/): For training and running the model.
- [Gradio](https://gradio.app/): For building the chatbot interface.

Install all dependencies with:
```bash
pip install torch transformers gradio
```

---

## **Future Improvements**

- Add support for multi-turn conversations.
- Expand the training dataset for better generalization.
- Implement more advanced generation techniques (e.g., beam search or nucleus sampling).
- Deploy the chatbot as a web app using services like AWS, Azure, or Hugging Face Spaces.

---

## **Acknowledgments**

- [Hugging Face](https://huggingface.co/) for their powerful models and libraries.
- [Gradio](https://gradio.app/) for their easy-to-use interface framework.
- OpenAI for GPT-2.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

Feel free to contribute, report issues, or suggest improvements! 😊
