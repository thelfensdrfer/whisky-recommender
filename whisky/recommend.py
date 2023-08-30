import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

from whisky.database import get_connection, get_user_id_by_username, get_whiskies_by_id

VALID_MODELS = ["knn_v1"]

WHISKY_MIN_RATING_COUNT = 20
USER_MIN_RATING_COUNT = 5

logger = logging.getLogger(__name__)


def get_ratings() -> pd.DataFrame:
    """
    Get all user ratings for all whiskies. The data is filtered to only include whiskies with at least 10 ratings and
    users with at least 5 ratings. We also remove whiskies with a rating below 70, as there are very few of them.
    :return:
    """
    logger.debug("Getting ratings...")

    # Get all ratings for whiskies that have at least 3 ratings to prevent overfitting
    # noinspection PyTypeChecker
    return pd.read_sql_query(
        """
            SELECT user.id                 AS user_id,
                   whisky_rating.whisky_id AS whisky_id,
                   whisky_rating.rating    AS rating
            FROM whisky_rating
                     JOIN user ON user.username = whisky_rating.username
            WHERE whisky_id IN (SELECT whisky_id
                                FROM whisky_rating
                                GROUP BY whisky_id
                                HAVING COUNT(whisky_id) >= ?)
              AND user.username IN (SELECT username
                                    FROM whisky_rating
                                    GROUP BY username
                                    HAVING COUNT(username) >= ?)
            AND rating >= 70
    """,
        get_connection(),
        params=(WHISKY_MIN_RATING_COUNT, USER_MIN_RATING_COUNT),
    )


def get_whiskies_to_rate(
    exclude_username: str = "", min_availability: int = None, max_price: int = None
) -> pd.DataFrame:
    """
    Get all whiskies except the ones a user already rated.

    :param exclude_username: The username of the user to exclude whiskies for.
    :param min_availability: The minimum number of available shops for one whisky.
    :param max_price: The maximum price of the whiskies.
    :return:
    """
    logger.debug("Getting whiskies...")

    sql = """
    SELECT id as whisky_id
    FROM whisky
    WHERE 
        id NOT IN (
            SELECT whisky_id
            FROM whisky_rating
            WHERE username = ?
        )
        AND id IN (
            SELECT whisky_id
            FROM whisky_rating
            GROUP BY whisky_id
            HAVING COUNT(whisky_id) >= ?
        )
    """

    params = (
        exclude_username,
        WHISKY_MIN_RATING_COUNT,
    )

    if min_availability is not None:
        sql += " AND availability >= ?"
        params += (min_availability,)

    if max_price is not None:
        sql += " AND price <= ?"
        params += (max_price,)

    return pd.read_sql_query(
        sql,
        get_connection(),
        params=params,
    )


def get_whiskies() -> pd.DataFrame:
    """
    Get all whiskies and their average rating.

    :return:
    """
    logger.debug("Getting whiskies...")

    sql = """
    SELECT
        whisky.bottling_serie,
        whisky.category_id,
        whisky.bottler_id,
        whisky.age,
        whisky.bottled_year,
        ROUND(whisky.strength),
        ROUND(whisky.price),
        AVG(whisky_rating.rating) AS avg_rating
    FROM whisky
    JOIN whisky_rating ON whisky_rating.whisky_id = whisky.id
    WHERE 
        whisky.id IN (
            SELECT whisky_id
            FROM whisky_rating
            GROUP BY whisky_id
            HAVING COUNT(whisky_id) >= ?
        )
    GROUP BY 
        whisky.bottling_serie,
        whisky.category_id,
        whisky.bottler_id,
        whisky.age,
        whisky.bottled_year,
        whisky.strength,
        whisky.price
    """

    params = WHISKY_MIN_RATING_COUNT

    return pd.read_sql_query(
        sql,
        get_connection(),
        params=params,
    )


def plot_ratings() -> None:
    """
    Plot the rating distribution and the number of ratings per whisky and user.
    :return:
    """
    df = get_ratings()

    # Plot rating distribution
    plt.figure()
    plt.hist(df["rating"], bins=100)
    plt.title("Rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.show()

    # Plot number of ratings per whisky
    plt.figure()
    plt.hist(
        df.groupby("whisky_id").count()["rating"],
        bins=[0, 3, 10, 20, 50, 100, 250, 1000],
    )
    plt.title("Number of ratings per whisky")
    plt.xlabel("Number of ratings")
    plt.ylabel("Count")
    plt.show()

    # Plot number of ratings per user
    plt.figure()
    plt.hist(
        df.groupby("user_id").count()["rating"], bins=[0, 3, 10, 20, 50, 100, 250, 1000]
    )
    plt.title("Number of ratings per user")
    plt.xlabel("Number of ratings")
    plt.ylabel("Count")
    plt.show(block=True)


def get_model_path(version: str) -> str:
    """
    Get the path to the model file.
    :return:
    """
    current_path = Path(__file__).parent.parent
    return f"{current_path}/model_{version}.p"


def train_ratings_knn_v1():
    """
    Train a KNN model to predict the rating of a whisky based on the ratings of other whiskies.
    :return:
    """
    logger.info("Preparing data...")

    df = get_ratings()

    x = df.drop(["rating"], axis=1)
    y = df["rating"]
    # Normalize ratings to values from 0 to 1
    y = (y - y.min()) / (y.max() - y.min())

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
    )

    logger.info("Testing different k values...")
    scores = []
    neighbors = np.arange(1, 20, 2)
    for k in neighbors:
        model = KNeighborsRegressor(n_neighbors=k, n_jobs=-1).fit(x_train, y_train)
        score = cross_val_score(
            model, x_train, y_train, cv=10, scoring="neg_mean_squared_error"
        )
        logger.debug(f"Score for k={k}: {score.mean()}")
        scores.append(score.mean())

    logger.info(f"Optimal k: {neighbors[np.argmax(scores)]}")

    # Saving model with optimal k for later use
    model = KNeighborsRegressor(
        n_neighbors=neighbors[np.argmax(scores)], n_jobs=-1
    ).fit(x_train, y_train)
    score = cross_val_score(
        model, x_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    logger.info(f"Score mean: {score.mean()}")
    logger.info(f"Score std: {score.std()}")
    logger.info(f"Score rmse: {np.sqrt(-score.mean())}")

    logger.info("Saving model...")
    pickle.dump(model, open(get_model_path("knn_v1"), "wb"))

    logger.info("Done.")


def train_whiskies_knn_v1():
    """
    Train a KNN model to predict the rating of a whisky based on the ratings of other whiskies.
    :return:
    """
    logger.info("Preparing data...")

    df = get_whiskies()

    x = df.drop(["rating"], axis=1)
    y = df["rating"]
    # Normalize ratings to values from 0 to 1
    y = (y - y.min()) / (y.max() - y.min())

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
    )

    logger.info("Testing different k values...")
    scores = []
    neighbors = np.arange(1, 20, 2)
    for k in neighbors:
        model = KNeighborsRegressor(n_neighbors=k, n_jobs=-1).fit(x_train, y_train)
        score = cross_val_score(
            model, x_train, y_train, cv=10, scoring="neg_mean_squared_error"
        )
        logger.debug(f"Score for k={k}: {score.mean()}")
        scores.append(score.mean())

    logger.info(f"Optimal k: {neighbors[np.argmax(scores)]}")

    # Plot k/scores
    plt.figure()
    plt.plot(neighbors, scores)
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.show(block=True)

    # Saving model with optimal k for later use
    model = KNeighborsRegressor(
        n_neighbors=neighbors[np.argmax(scores)], n_jobs=-1
    ).fit(x_train, y_train)
    score = cross_val_score(
        model, x_train, y_train, cv=10, scoring="neg_mean_squared_error"
    )
    logger.info(f"Score mean: {score.mean()}")
    logger.info(f"Score std: {score.std()}")
    logger.info(f"Score rmse: {np.sqrt(-score.mean())}")

    logger.info("Saving model...")
    pickle.dump(model, open(get_model_path("knn_v1"), "wb"))

    logger.info("Done.")


def recommend_whiskies(
    username: str,
    model: str = "knn_v1",
    top_n: int = 10,
    min_availability: int = None,
    max_price: int = None,
) -> None:
    """
    Recommend whiskies for a user.
    :param username: The username of the user to recommend whiskies for.
    :param model: The model to use for predictions.
    :param top_n: The number of whiskies to recommend.
    :param min_availability: The minimum number of available shops for one whisky.
    :param max_price: The maximum price of the whiskies.
    :return:
    """
    logger.debug("Loading model...")

    # Check if model is valid
    if model not in VALID_MODELS:
        raise ValueError(
            f"Model {model} is not valid. Valid models: {', '.join(VALID_MODELS)}"
        )

    # Check if file exists
    if not Path(get_model_path(model)).is_file():
        logger.info(f"Model {model} does not exist, training it...")
        if model == "knn_v1":
            train_ratings_knn_v1()

    user_id = get_user_id_by_username(username)

    # Load model
    model = pickle.load(open(get_model_path(model), "rb"))

    # Create predictions
    logger.info("Creating predictions...")
    whiskies = get_whiskies_to_rate(
        exclude_username=username,
        min_availability=min_availability,
        max_price=max_price,
    )
    whiskies["user_id"] = user_id
    # Reorder columns so that the order is the same as in the training data
    whiskies = whiskies[["user_id", "whisky_id"]]
    prediction = model.predict(whiskies)

    # Order predictions by rating
    whiskies["rating"] = prediction
    whiskies = whiskies.sort_values(by=["rating"], ascending=False)

    # Get top 10 whiskies
    whiskies_sorted_top = whiskies.head(top_n)

    # Get whisky names
    whiskies_top_n = get_whiskies_by_id(whiskies_sorted_top["whisky_id"].tolist())
    logger.info(f"Top {top_n} whiskies for user {username}:")
    i = top_n
    for whisky in reversed(whiskies_top_n):
        rating = whiskies_sorted_top[whiskies_sorted_top["whisky_id"] == whisky["id"]][
            "rating"
        ].values[0]
        logger.info(f"{i}: {whisky['name']} (#WB{whisky['id']}) [{round(rating*100)}%]")
        i = i - 1
