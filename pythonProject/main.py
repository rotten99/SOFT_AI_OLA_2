import pandas as pd
from LinearRegression import perform_linear_regression
from InputGetter import get_user_data
from TreeRegression import perform_decision_tree_regression
from RandomForrestRegression import perform_random_forest_regression


def main():
    df = pd.read_csv('F1DriversDataset.csv')

    while True:
        print("\nWhat regression model would you like to use?")
        print("-----------------")
        print("1. Linear regression")
        print("2. Decision tree regression")
        print("3. Random forrest regression")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            print("You've selected Linear regression")
            userData = get_user_data()
            perform_linear_regression(df,userData[0],userData[1],userData[2])
            # Add functionality for Option 1 here
        elif choice == '2':
            print("You've selected Decision tree regression")
            get_user_data()
            perform_decision_tree_regression(df,userData[0],userData[1],userData[2])
            # Add functionality for Option 2 here
        elif choice == '3':
            print("You've selected Random forrest regression")
            get_user_data()
            perform_random_forest_regression(df,userData[0],userData[1],userData[2])
            # Add functionality for Option 3 here
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid input. Please enter a number between 1 and 4.")


if __name__ == "__main__":
    main()
