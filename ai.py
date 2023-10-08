import os
import speech_recognition as sr
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import wikipedia

class SelfImprovingAI:
    def __init__(self, model_path="self_improving_model"):
        self.model_path = model_path
        self.training_data = []
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.context = ""

    def train_model(self):
        # Use the training data to update or retrain the model
        # In a real-world scenario, this would involve fine-tuning the model
        pass

    def save_model(self):
        # Save the current model state
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

    def load_model(self):
        # Load the model from the specified path
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)

    def auto_update(self, new_data):
        # In a real-world scenario, this might involve fetching new data from a data source
        # Here, we'll simulate it by adding the new data
        self.training_data.append({"input": "", "output": new_data, "feedback": "positive"})

        # Retrain the model with the updated data
        self.train_model()

    def generate_response(self, user_input):
        # Combine current input with context
        input_text = self.context + user_input

        # Tokenize user input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generate a response
        output = self.model.generate(input_ids, max_length=50, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Gather feedback on the response
        user_feedback = input("Was the AI response helpful? (yes/no): ")

        # Include feedback in the training data
        self.training_data.append({"input": input_text, "output": response, "feedback": user_feedback})

        # If the response contains a question, ask for more information
        if "?" in response:
            additional_info = input("I'm not sure I understand. Could you provide more information or context? ")
            self.auto_update(additional_info)

        # If the user asks for information, fetch it from Wikipedia
        if "tell me about" in user_input.lower():
            topic = user_input.lower().replace("tell me about", "").strip()
            try:
                summary = wikipedia.summary(topic, sentences=1)
                print("Wikipedia Summary:", summary)
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"Ambiguous search term. Suggestions: {', '.join(e.options)}")
            except wikipedia.exceptions.PageError:
                print("Sorry, I couldn't find information on that topic.")

        # Retrain the model periodically with the updated training data
        self.train_model()

        # Update context for the next turn
        self.context = input_text + " " + response

        return response

    def speech_to_text(self):
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            print("Processing audio...")
            user_input = recognizer.recognize_google(audio)
            print("You said:", user_input)
            return user_input
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return ""

# Example usage
if __name__ == "__main__":
    ai_system = SelfImprovingAI()

    # Load the model if it exists
    if os.path.exists(ai_system.model_path):
        ai_system.load_model()
    else:
        # Train the model initially with some default data
        ai_system.train_model()

    while True:
        print("1. Type your question")
        print("2. Speak your question")
        print("3. Exit")

        choice = input("Select an option (1/2/3): ")

        if choice == "1":
            user_input = input("User: ")
        elif choice == "2":
            user_input = ai_system.speech_to_text()
        elif choice == "3":
            break
        else:
            print("Invalid option. Please try again.")
            continue

        ai_response = ai_system.generate_response(user_input)
        print("AI:", ai_response)

        # Save the model periodically
        if len(ai_system.training_data) % 5 == 0:
            ai_system.save_model()
