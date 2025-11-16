import sys
from model_training import train_model
from simulation import simulate_betting
from visualization import visualize_results


def main():
    while True:
        print("\n*** Football Bets Simulator ***")
        print("1. Train Model")
        print("2. Simulate Bets")
        print("3. Visualize Results")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            print("\nTraining model...")
            train_model()
        elif choice == "2":
            print("\nSimulating betting...")
            try:
                bet_amount = float(input("Enter bet amount (default=10): ") or "10")
                threshold = float(input("Enter confidence threshold (default=0.6): ") or "0.6")
                simulate_betting(bet_amount=bet_amount, threshold=threshold)
            except ValueError:
                print("Invalid input. Please enter numeric values.")
        elif choice == "3":
            print("\nVisualizing model accuracy...")
            visualize_results()
        elif choice == "4":
            print("\nExiting program")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
