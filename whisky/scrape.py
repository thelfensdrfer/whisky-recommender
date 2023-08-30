import asyncio
import logging
import re
import sys
import time

import aiohttp
import requests
from aiohttp import ClientSession
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag

from whisky.database import (
    insert_whisky_into_database,
    get_or_insert_distillery,
    insert_users_into_database,
    get_all_users,
    insert_ratings_into_database,
)


BASE_URL = "https://www.whiskybase.com/whiskies/distilleries"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/116.0"
BROWSER_FINGERPRINT = "703032580ab93edd781941372ee551f1"
COOKIES = {}
MAX_CONNECTIONS = 10

logger = logging.getLogger(__name__)


def login(username: str, password: str) -> None:
    """Login to whiskybase.com"""
    logger.debug("Logging in to whiskybase.com...")

    response = requests.get(
        "https://www.whiskybase.com/account/login", {"User-Agent": USER_AGENT}
    )
    soup = BeautifulSoup(response.text, "html.parser")

    csrf_token = soup.select_one('.login-form input[name="_token"]').get("value")
    udf = BROWSER_FINGERPRINT

    xsrf_token = response.cookies["XSRF-TOKEN"]
    barrel_token = response.cookies["barrel"]

    data = {
        "_token": csrf_token,
        "username": ["", username],
        "p": "",
        "udf": udf,
        "password": password,
    }

    response = requests.post(
        "https://www.whiskybase.com/account/login",
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://www.whiskybase.com",
            "Referer": "https://www.whiskybase.com/account/login",
        },
        cookies={
            "XSRF-TOKEN": xsrf_token,
            "barrel": barrel_token,
        },
    )

    if not response.ok:
        raise Exception(
            f"Failed to login to whiskybase.com (Error {response.status_code})!"
        )

    logger.debug("Successfully logged in to whiskybase.com")
    COOKIES["barrel"] = response.cookies["barrel"]


def dl_to_dict(dl: BeautifulSoup, raw=False) -> dict:
    """
    Convert a dl tag to a dictionary

    :param dl: The dl tag
    :param raw: Whether to return the raw text or the BeautifulSoup object
    """
    return {
        dt.text.strip(): dd.text.strip() if not raw else dd
        for dt, dd in zip(dl.select("dt"), dl.select("dd"))
    }


def insert_whisky(distillery_name: str, html: str, url: str) -> int:
    """
    Insert a whisky into the database

    :param distillery_name: The name of the distillery
    :param html: The HTML of the whisky page
    :param url: The URL of the whisky page
    """
    logger.debug("Inserting whisky into database...")

    soup = BeautifulSoup(html, "html.parser")
    detail_list = soup.select_one("#whisky-details dl")
    if not detail_list:
        logger.warning(f'No details found for {soup.select_one("h1").text.strip()}')
        return 0

    details = dl_to_dict(detail_list)
    details_raw = dl_to_dict(detail_list, raw=True)
    tags = soup.select("#whisky-details dl img")
    tasting_notes = soup.select('.tastingtags li a:not([data-num=""])')

    whisky_id = int(re.search(r"\d+", details["Whiskybase ID"]).group())
    category = "Category" in details and details["Category"] or None
    bottler = "Bottler" in details and details["Bottler"] or None
    bottling_serie = "Bottling serie" in details and details["Bottling serie"] or None
    age = "Stated Age" in details and details["Stated Age"] or None
    age_years = int(re.search(r"\d+", age).group()) if age else None
    bottled: str or None = "Bottled" in details and details["Bottled"] or None
    bottled_year = None
    if bottled and "." in bottled:
        bottled_date_parts = bottled.split(".")
        bottled_year_filter = list(filter(lambda x: len(x) == 4, bottled_date_parts))
        if bottled_year_filter:
            bottled_year = int(bottled_year_filter[0])
    strength = "Strength" in details and details["Strength"] or None
    strength_vol = (
        float(re.search(r"\d+[.\d+]*", strength).group()) if strength else None
    )
    size = "Size" in details and details["Size"] or None
    size_ml = int(re.search(r"\d+", size).group()) if size else None
    name = soup.select_one(".name header h1").text.strip()
    # Replace line breaks, tabs and multiple spaces with a single space
    name = re.sub(r"[\n\t]+", " ", name)
    price = None
    price_tag = soup.select_one(".block-price")
    if price_tag:
        price = price_tag.text.strip()
        # Get the float value from the price
        price = float(re.search(r"\d+[.\d+]*", price).group())
    availability = None
    availability_tag = soup.select_one("#panel-shoplinks p:first-child")
    if availability_tag:
        availability = availability_tag.text.strip()
        availability = int(re.search(r"\d+", availability).group())

    distilleries = []
    if "Distilleries" in details_raw:
        for distillery in details_raw["Distilleries"].select("a"):
            distilleries.append(
                {
                    "name": distillery.text.strip(),
                    "url": distillery.get("href"),
                }
            )
    elif "Distillery" in details_raw:
        distillery = details_raw["Distillery"].select_one("a")
        distilleries.append(
            {
                "name": distillery.text.strip(),
                "url": distillery.get("href"),
            }
        )

    whisky_details = {
        "id": whisky_id,
        "name": name,
        "distilleries": distilleries,
        "category": category,
        "bottler": bottler,
        "bottling_serie": bottling_serie,
        "age": age_years,
        "bottled_year": bottled_year,
        "strength": strength_vol,
        "size": size_ml,
        "url": url,
        "price": price,
        "availability": availability,
        "tags": [tag.get("title").strip() for tag in tags],
        "tasting_notes": [
            {
                "id": int(tasting_note.get("data-id").strip()),
                "votes": int(tasting_note.get("data-num").strip()),
                "name": tasting_note.select_one(".tag-name").text.strip(),
            }
            for tasting_note in tasting_notes
        ],
    }

    return insert_whisky_into_database(whisky_details)


async def scrape_distillery(
    distillery: Tag, session: ClientSession, pbar: tqdm
) -> None:
    href = distillery.select_one("td.clickable a").get("href")

    distillery_response = await session.get(href)

    if not distillery_response.ok:
        logger.error(
            f"Failed to get whisky list from distillery {href} (Error {distillery_response.status})!"
        )
        return

    distillery_name = distillery.select_one("td.clickable a").text.strip()
    distillery_country = distillery.select_one("td:nth-child(2)").text.strip()
    get_or_insert_distillery(distillery_name, href, distillery_country)

    whiskies = BeautifulSoup(await distillery_response.text(), "html.parser").select(
        "table.whiskytable a.clickable"
    )
    for whisky in whiskies:
        href = whisky.get("href")

        whisky_response = await session.get(href)

        if not whisky_response.ok and whisky_response.status != 404:
            # 404 errors are expected for some whiskies
            logger.error(
                f"Failed to get whisky {href} from {distillery_name} (Error {whisky_response.status})"
            )
            continue

        try:
            insert_whisky(distillery_name, await whisky_response.text(), href)
        except Exception:
            logger.exception(f"Failed to insert whisky {href}!")

    pbar.update()


async def scrape_whiskies(username: str, password: str, profile: bool):
    """
    Scrape the distilleries and all of their whiskies.

    :param username: The username for the whiskybase.com website
    :param password: The password for the whiskybase.com website
    :param profile: Whether to only scrape a small percentage to profile and test the functionality.
    :return:
    """
    logger.debug("Scraping distillery list from %s", BASE_URL)

    start_time = time.time()

    # Get the HTML from the page
    # TODO: Currently not needed (only needed if we want to get the contents of the ratings)
    # login(username, password)
    response = requests.get(BASE_URL, headers={"User-Agent": USER_AGENT})

    if not response.ok:
        logger.error(
            f"Failed to get list of distilleries from {BASE_URL}: {response.text} (Error {response.status_code})"
        )
        return

    # Find the distillery list
    distilleries = BeautifulSoup(response.text, "html.parser").select(
        "table.whiskytable tbody tr"
    )
    if profile:
        distilleries = distilleries[:10]

    pbar = tqdm(distilleries, desc="Scraping distilleries")

    connector = aiohttp.TCPConnector(limit=MAX_CONNECTIONS)
    async with aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT},
        connector=connector,
    ) as session:
        await asyncio.gather(
            *[
                scrape_distillery(distillery, session, pbar)
                for distillery in distilleries
            ]
        )

    end_time = time.time()
    logger.debug("Scraped whiskies in %s seconds", end_time - start_time)


async def scrape_users_for_char(char: str, pbar: tqdm, session: ClientSession) -> None:
    response = await session.get(
        f"https://www.whiskybase.com/contribute/community/member-list?search=null&chr={char}&country_id=&h=user.username,user.notes,generic.votes,generic.country"
    )
    if not response.ok:
        logger.error(
            f"Failed to get list of users for character {char} (Error {response.status})"
        )
        return

    soup = BeautifulSoup(await response.text(), "html.parser")

    usernames = []
    for member in soup.select(".whiskytable tbody tr"):
        has_votes = member.select_one("td:nth-child(4)").text.strip() != "-"
        if has_votes:
            profile_url = member.select_one("td:nth-child(2) a").get("href").strip()
            username = re.search(r"/profile/(.*)", profile_url).group(1)

            if not username:
                continue

            usernames.append(
                {
                    "username": username,
                    "country": member.select_one("td:nth-child(5) img")
                    .get("title")
                    .strip(),
                }
            )

    insert_users_into_database(usernames)
    pbar.update()


async def scrape_users(profile: bool):
    """
    Scrape the users which have at least one note or rating from whiskybase.com

    :param profile: Whether to only scrape a small percentage to profile and test the functionality.
    :return:
    """
    chars = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    if profile:
        chars = chars[:10]

    pbar = tqdm(chars, desc="Scraping users")

    connector = aiohttp.TCPConnector(limit=MAX_CONNECTIONS)
    async with aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT}, connector=connector
    ) as session:
        await asyncio.gather(
            *[scrape_users_for_char(char, pbar, session) for char in chars]
        )


async def scrape_ratings_for_user(
    username: str, pbar: tqdm, session: ClientSession
) -> None:
    url = f"https://www.whiskybase.com/profile/{username}/lists/ratings"
    try:
        response = await session.get(url)
    except Exception:
        logger.exception(f"Failed to get ratings for user {url}!")
        return

    if not response.ok and response.status != 500:
        # 500 errors are expected for some users because their profile is hidden?
        logger.error(f"Failed to get ratings for user {url}: (Error {response.status})")
        return

    soup = BeautifulSoup(await response.text(), "html.parser")

    ratings = []
    for row in soup.select(".whiskytable tbody tr"):
        rating = row.select_one("td:nth-child(10)").text.strip()
        if rating != "-":
            whisky_url = row.select_one("td:nth-child(2) a").get("href")
            whisky_id = int(re.search(r"/\d+/", whisky_url).group().replace("/", ""))
            rating = int(rating)

            ratings.append(
                {
                    "username": username,
                    "whisky_id": whisky_id,
                    "rating": rating,
                }
            )

    insert_ratings_into_database(ratings)
    pbar.update()


async def scrape_ratings(profile: bool):
    """
    Scrape the ratings for all users.

    :param profile: Whether to only scrape a small percentage to profile and test the functionality.
    :return:
    """
    users = get_all_users()
    if profile:
        users = users[:1000]
    pbar = tqdm(users, desc="Scraping ratings")

    connector = aiohttp.TCPConnector(limit=MAX_CONNECTIONS)
    async with aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT},
        connector=connector,
        # We disable the timeout because the ratings page can take a long time to load
        read_timeout=None,
    ) as session:
        await asyncio.gather(
            *[scrape_ratings_for_user(username, pbar, session) for username in users]
        )
