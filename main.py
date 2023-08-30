import asyncio
import logging
import click
from whisky.database import (
    connect_database,
    create_database,
    close_database,
    truncate_users_and_ratings,
)
from whisky.scrape import scrape_whiskies, scrape_users, scrape_ratings
from whisky.recommend import (
    train_ratings_knn_v1,
    plot_ratings,
    recommend_whiskies,
    VALID_MODELS,
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--reset", is_flag=True, help="Delete database and scrape data again.")
@click.option(
    "--ratings", is_flag=True, help="Reset and scrape only users and ratings."
)
@click.option(
    "--profile",
    is_flag=True,
    help="Only scrape a small percentage to profile and test the functionality.",
)
@click.option("--username", help="Your username for the whiskybase.com website.")
@click.option("--password", help="Your password for the whiskybase.com website.")
def scrape(
    username: str,
    password: str,
    reset: bool = False,
    profile: bool = False,
    ratings: bool = False,
):
    if reset and not click.confirm(
        "Are you sure you want to delete the database and scrape again?"
    ):
        return

    connect_database()
    create_database(reset=reset)

    if reset:
        asyncio.run(scrape_whiskies(username, password, profile))
        asyncio.run(scrape_users(profile))
        asyncio.run(scrape_ratings(profile))

    if ratings:
        truncate_users_and_ratings()

        asyncio.run(scrape_users(profile))
        asyncio.run(scrape_ratings(profile))

    close_database()

    logger.info("Done.")


@cli.command()
def train():
    connect_database()
    train_ratings_knn_v1()
    close_database()


@cli.command()
def plot():
    connect_database()
    plot_ratings()
    close_database()


@cli.command()
@click.option("--username", help="Your username for the whiskybase.com website.")
@click.option(
    "--model",
    help=f"The model to use for predictions. Valid models: {', '.join(VALID_MODELS)}.",
    default="knn_v1",
)
@click.option("--top", help="The number of whiskies to recommend.", default=10)
@click.option(
    "--min_availability", help="The minimum number of available shops for one whisky."
)
@click.option("--max_price", help="The maximum price of the whiskies.")
def predict(username: str, model: str, top: int, min_availability: int, max_price: int):
    connect_database()
    recommend_whiskies(
        username,
        model,
        top_n=top,
        min_availability=min_availability,
        max_price=max_price,
    )
    close_database()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cli()
