def get_user_data() -> list:

    while True:
        champion_input = input("Do you want champion level drivers (yes/no): ").strip().lower()
        if champion_input in ['yes', 'y']:
            isChampion = True
            break
        elif champion_input in ['no', 'n']:
            isChampion = False
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    # Get and validate integer input for race_starts
    while True:
        race_starts_input = input("Enter the number of race starts: ").strip()
        try:
            race_starts = int(race_starts_input)
            break
        except ValueError:
            print("Invalid input. Please enter an integer value.")

    # Get and validate integer input for season
    while True:
        season_input = input("Enter the season year (e.g., 2018): ").strip()
        try:
            season = int(season_input)
            if 1950 <= season <= 2021:
                break
            else:
                print("Season year must be between 1950 and 2021.")
        except ValueError:
            print("Invalid input. Please enter a valid year as an integer.")

    return [isChampion, race_starts, str(season)]