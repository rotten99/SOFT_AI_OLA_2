from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def perform_linear_regression(df,isChampion, Race_Starts, season):

    # Filter the dataset based on champion status.
    if isChampion:
        subset = df[df['Champion'] == True]
    else:
        subset = df[df['Champion'] == False]

    # Further filter to include only drivers whose 'Seasons' string contains the given season.
    subset = subset[subset['Seasons'].str.contains(season)]

    # Extract independent variable (X) and dependent variable (y).
    X = subset[['Race_Starts']].values
    y = subset['Points'].values

    # Initialize and fit the Linear Regression model with intercept fixed to 0.
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Calculate the coefficient of determination (R^2).
    r2 = model.score(X, y)

    # Prepare the input for prediction.
    if isinstance(Race_Starts, (int, float)):
        X_new = np.array([[Race_Starts]])
    elif isinstance(Race_Starts, list) or isinstance(Race_Starts, np.ndarray):
        X_new = np.array([[x] for x in Race_Starts])
    else:
        raise ValueError("Race_Starts must be a number or a list/array of numbers.")

    # Predict Points for the provided Race_Starts value(s).
    predicted_points = model.predict(X_new)

    # Print model details.
    group = "Champions" if isChampion else "Non-Champions"
    print(f"Linear Regression Model for {group} in season {season}:")
    print("  Coefficient (slope):", model.coef_[0])
    print("  Intercept:", model.intercept_)
    print("  Coefficient of Determination (R^2):", r2)
    print("  Predicted Points for Race_Starts =", Race_Starts, ":", predicted_points)

    # Compute predictions for the subset data to calculate percentages.
    predicted_subset = model.predict(X)
    count_above = np.sum(y > predicted_subset)
    count_below = np.sum(y < predicted_subset)
    total = len(y)
    perc_above = (count_above / total) * 100 if total > 0 else 0
    perc_below = (count_below / total) * 100 if total > 0 else 0

    # Create a range of Race_Starts values for a smooth regression line.
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(X_range)

    # Plot the regression graph.
    plt.figure(figsize=(8, 6))
    if isChampion:
        plt.scatter(X, y, color='blue', label='Data Points')
        line_color = 'red'
    else:
        plt.scatter(X, y, color='green', label='Data Points')
        line_color = 'orange'

    plt.plot(X_range, y_range, color=line_color, linewidth=2, label='Regression Line')
    plt.xlabel("Race Starts")
    plt.ylabel("Points")
    plt.title(f"Linear Regression for {group} in Season {season}")
    plt.legend()
    plt.grid(True)
    plt.text(0.05, 0.95, f"Above: {perc_above:.1f}%\nBelow: {perc_below:.1f}%",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

    return model, predicted_points