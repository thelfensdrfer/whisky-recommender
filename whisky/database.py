import logging
import sqlite3
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)
conn: sqlite3.Connection
CACHE = {}


def database_path() -> str:
    """
    Get the path to the database file.
    :return:
    """
    current_path = Path(__file__).parent.parent
    return f"{current_path}/whisky.db"


def connect_database() -> None:
    global conn

    logger.debug("Connecting to database...")

    conn = sqlite3.connect(database_path())
    conn.row_factory = sqlite3.Row


def create_database(reset: bool = False) -> None:
    """
    Create a new sqlite database with the schema defined in the schema.sql file.

    :param reset: If True, delete the existing database before creating a new one.
    """
    logger.debug("Creating database...")

    global conn

    current_path = Path(__file__).parent.parent
    schema_file = f"{current_path}/schema.sql"

    db_file = Path(database_path())
    if db_file.is_file():
        logger.debug("Database already exists.")
        if reset:
            close_database()

            # Database is deleted and recreated.
            logger.info("Deleting existing database...")
            db_file.unlink()
            logger.debug("Database deleted.")

            connect_database()
        else:
            # Database already exists.
            return

    # Database was deleted or didn't exist, so we apply the migration again
    cursor = conn.cursor()

    with open(schema_file) as f:
        cursor.executescript(f.read())

    conn.commit()
    logger.debug("Database created.")


def close_database() -> None:
    """
    Close the database connection.
    """
    logger.debug("Closing database...")
    global conn

    if conn:
        conn.close()
        logger.debug("Database closed.")
    else:
        logger.debug("Database not open.")


def get_or_insert_category(name: str or None) -> Union[int, None]:
    """
    Get the id of a category from the database, or insert it if it doesn't exist.

    :param name: The name of the category.
    :return: The id of the category.
    """
    if not name:
        return None

    global conn

    if "category" not in CACHE:
        CACHE["category"] = {}

    if name in CACHE["category"]:
        return CACHE["category"][name]

    cursor = conn.cursor()
    cursor.execute("SELECT id FROM category WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        CACHE["category"][name] = row[0]
        return row[0]
    else:
        cursor.execute("INSERT INTO category (name) VALUES (?)", (name,))
        conn.commit()

        CACHE["category"][name] = cursor.lastrowid
        return cursor.lastrowid


def get_or_insert_bottler(name: str or None) -> Union[int, None]:
    """
    Get the id of a bottler from the database, or insert it if it doesn't exist.

    :param name: The name of the bottler.
    :return: The id of the bottler.
    """
    if not name:
        return None

    global conn

    if "bottler" not in CACHE:
        CACHE["bottler"] = {}

    if name in CACHE["bottler"]:
        return CACHE["bottler"][name]

    cursor = conn.cursor()
    cursor.execute("SELECT id FROM bottler WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        CACHE["bottler"][name] = row[0]
        return row[0]
    else:
        cursor.execute("INSERT INTO bottler (name) VALUES (?)", (name,))
        conn.commit()

        CACHE["bottler"][name] = cursor.lastrowid
        return cursor.lastrowid


def get_or_insert_distillery(name: str, url: str, country: str = None) -> int:
    """
    Get the id of a distillery from the database, or insert it if it doesn't exist.

    :param name: The name of the distillery.
    :param url: The url of the distillery.
    :param country: The country of the distillery.
    :return: The id of the distillery.
    """
    if "distillery" not in CACHE:
        CACHE["distillery"] = {}

    if name in CACHE["distillery"]:
        return CACHE["distillery"][name]

    cursor = conn.cursor()
    cursor.execute("SELECT id FROM distillery WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        CACHE["distillery"][name] = row[0]
        return row[0]
    else:
        cursor.execute(
            "INSERT INTO distillery (name, url, country) VALUES (?, ?, ?)",
            (
                name,
                url,
                country,
            ),
        )
        conn.commit()

        CACHE["distillery"][name] = cursor.lastrowid
        return cursor.lastrowid


def get_or_insert_tag(name: str) -> int:
    """
    Get the id of a tag from the database, or insert it if it doesn't exist.

    :param name: The name of the tag.
    :return: The id of the tag.
    """
    global conn

    if "tag" not in CACHE:
        CACHE["tag"] = {}

    if name in CACHE["tag"]:
        return CACHE["tag"][name]

    cursor = conn.cursor()
    cursor.execute("SELECT id FROM tag WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        CACHE["tag"][name] = row[0]
        return row[0]
    else:
        cursor.execute("INSERT INTO tag (name) VALUES (?)", (name,))
        conn.commit()

        CACHE["tag"][name] = cursor.lastrowid
        return cursor.lastrowid


def get_or_insert_tasting_note(
    note_id: int, name: str, votes: int
) -> list[int, str, int]:
    """
    Get the id of a tasting note from the database, or insert it if it doesn't exist.

    :param note_id: The id of the tasting note.
    :param name: The name of the tasting note.
    :param votes: The number of votes for the tasting note.
    :return: The id of the tasting note.
    """
    global conn

    if "tasting_note" not in CACHE:
        CACHE["tasting_note"] = {}

    if note_id in CACHE["tasting_note"]:
        return CACHE["tasting_note"][note_id]

    cursor = conn.cursor()
    cursor.execute("SELECT id FROM tasting_note WHERE id = ?", (note_id,))
    row = cursor.fetchone()
    if row:
        CACHE["tasting_note"][note_id] = [note_id, name, votes]
        return [note_id, name, votes]
    else:
        cursor.execute(
            "INSERT INTO tasting_note (id, name) VALUES (?, ?)",
            (
                note_id,
                name,
            ),
        )
        conn.commit()

        CACHE["tasting_note"][note_id] = [note_id, name, votes]
        return [note_id, name, votes]


def get_or_insert_user(rating: dict) -> str:
    """
    Get the username of a user from the database, or insert it if it doesn't exist.

    :param rating:
    :return:
    """
    global conn

    rating["user"] = rating["user"].replace("https://www.whiskybase.com/profile/", "")

    if "user" not in CACHE:
        CACHE["user"] = {}

    if rating["user"] in CACHE["user"]:
        return CACHE["user"][rating["user"]]

    cursor = conn.cursor()
    cursor.execute("SELECT username FROM user WHERE username = ?", (rating["user"],))
    row = cursor.fetchone()
    if not row:
        cursor.execute("INSERT INTO user (username) VALUES (?)", (rating["user"],))
        conn.commit()

    CACHE["user"][rating["user"]] = rating["user"]
    return rating["user"]


def insert_whisky_into_database(details: dict) -> int:
    """
    Insert a whisky into the database.
    :param details:
    :return: Whisky id
    """

    global conn
    cursor = conn.cursor()

    distilleries = []
    for distillery in details["distilleries"]:
        distilleries.append(
            get_or_insert_distillery(distillery["name"], distillery["url"])
        )

    category = get_or_insert_category(details["category"])

    bottler = get_or_insert_bottler(details["bottler"])

    tags = []
    for tag in details["tags"]:
        tags.append(get_or_insert_tag(tag))

    tasting_notes = []
    for tasting_note in details["tasting_notes"]:
        # TODO: Votes are note saved correctly?
        tasting_notes.append(
            get_or_insert_tasting_note(
                tasting_note["id"], tasting_note["name"], tasting_note["votes"]
            )
        )

    try:
        cursor.execute(
            "INSERT INTO whisky (id, name, category_id, bottler_id, bottling_serie, age, bottled_year, strength, size, url, price, availability) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                details["id"],
                details["name"],
                category,
                bottler,
                details["bottling_serie"],
                details["age"],
                details["bottled_year"],
                details["strength"],
                details["size"],
                details["url"],
                details["price"],
                details["availability"],
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        logger.debug(f"Whisky with id {details['id']} already exists in the database.")
        return details["id"]

    whisky_id = cursor.lastrowid

    cursor.executemany(
        "INSERT INTO whisky_distillery (whisky_id, distillery_id) VALUES (?, ?)",
        [(whisky_id, distillery) for distillery in distilleries],
    )
    conn.commit()

    cursor.executemany(
        "INSERT INTO whisky_tags (whisky_id, tag_id) VALUES (?, ?)",
        [(whisky_id, tag) for tag in tags],
    )
    conn.commit()

    cursor.executemany(
        "INSERT INTO whisky_tasting_note (whisky_id, tasting_note_id, votes) VALUES (?, ?, ?)",
        [
            (whisky_id, tasting_note[0], tasting_note[2])
            for tasting_note in tasting_notes
        ],
    )
    conn.commit()

    return whisky_id


def insert_users_into_database(users: list[dict]) -> None:
    """
    Insert a user into the database.
    :param users:
    :return:
    """

    global conn
    cursor = conn.cursor()

    cursor.executemany(
        "INSERT INTO user (username, country) VALUES (?, ?) ON CONFLICT DO NOTHING",
        [(user["username"], user["country"]) for user in users],
    )
    conn.commit()


def get_all_users() -> list[str]:
    """
    Get all users from the database.
    :return:
    """
    global conn
    cursor = conn.cursor()

    cursor.execute("SELECT username FROM user")
    return [row[0] for row in cursor.fetchall()]


def insert_ratings_into_database(ratings: list[dict]) -> None:
    """
    Insert a rating into the database.
    :param ratings:
    :return:
    """

    global conn
    cursor = conn.cursor()

    cursor.executemany(
        "INSERT INTO whisky_rating (username, whisky_id, rating) VALUES (?, ?, ?)",
        [
            (
                rating["username"],
                rating["whisky_id"],
                rating["rating"],
            )
            for rating in ratings
        ],
    )
    conn.commit()


def get_connection() -> sqlite3.Connection:
    """
    Get the database connection.
    :return:
    """
    global conn

    return conn


def truncate_users_and_ratings() -> None:
    """
    Delete all users from the database.
    :return:
    """
    global conn
    cursor = conn.cursor()

    cursor.execute("DELETE FROM whisky_rating")
    cursor.execute("DELETE FROM user")
    conn.commit()


def get_user_id_by_username(username: str) -> int:
    """
    Get the user id by username.
    :param username:
    :return:
    """
    global conn
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM user WHERE username = ?", (username,))
    row = cursor.fetchone()
    if row:
        return row[0]

    raise ValueError(f"User with username {username} not found.")


def get_whiskies_by_id(whisky_ids: list[int]) -> list:
    """
    Get whiskies by id.
    :return:
    """
    global conn
    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT *
        FROM whisky
        WHERE id IN ({",".join(["?"] * len(whisky_ids))})
        """,
        whisky_ids,
    )
    return [dict(row) for row in cursor.fetchall()]
