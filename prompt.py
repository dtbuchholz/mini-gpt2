import os
from generate import load_model, generate_text, GPTConfig
from dotenv import load_dotenv

load_dotenv()

# Model path will default to the latest checkpoint in the log directory
TESTING_MODEL_PATH = os.environ.get("TESTING_MODEL_PATH", "log/")


def main():
    print("Loading model...")
    model = load_model(TESTING_MODEL_PATH)  # Use your latest checkpoint
    print("\nTiny GPT Chatbot")
    print("------------------------")
    print("Type 'quit' to exit")
    print("Type 'temp X' to change temperature (0.1-2.0)")
    print("Type 'tokens X' to change max tokens (1-1000)")

    # Default settings
    temperature = 0.8
    max_tokens = 200

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for commands
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            elif user_input.lower().startswith("temp "):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"Temperature set to {temperature}")
                    else:
                        print("Temperature must be between 0.1 and 2.0")
                except:
                    print("Invalid temperature format")
                continue
            elif user_input.lower().startswith("tokens "):
                try:
                    new_tokens = int(user_input.split()[1])
                    if 1 <= new_tokens <= 1000:
                        max_tokens = new_tokens
                        print(f"Max tokens set to {max_tokens}")
                    else:
                        print("Max tokens must be between 1 and 1000")
                except:
                    print("Invalid tokens format")
                continue

            # Generate response
            if not user_input:
                continue

            print("\nBot: ", end="", flush=True)
            response = generate_text(
                model, user_input, max_tokens=max_tokens, temperature=temperature
            )
            print(response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
