import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree


def perform_decision_tree_regression(df,isChampion, Race_Starts, season, max_depth=None):

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

    # Initialize and fit the Decision Tree Regressor.
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
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

    # Print out the model details.
    group = "Champions" if isChampion else "Non-Champions"
    print(f"Decision Tree Regression Model for {group} in season {season}:")
    print("  Coefficient of Determination (R^2):", r2)
    print("  Predicted Points for Race_Starts =", Race_Starts, ":", predicted_points)

    # Plot the regression results (piecewise constant predictions) along with the data points.
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue' if isChampion else 'green', label='Data Points')
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(X_range)
    plt.plot(X_range, y_range, color='red', linewidth=2, label='Decision Tree Prediction')
    plt.xlabel("Race_Starts")
    plt.ylabel("Points")
    plt.title(f"Decision Tree Regression for {group} in Season {season}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the decision tree structure.
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=["Race_Starts"], rounded=True)
    plt.title(f"Decision Tree Structure for {group} in Season {season}")
    plt.show()

    return model, predicted_points