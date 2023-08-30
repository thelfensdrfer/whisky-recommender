## Usage

1. Install dependencies: `poetry install`
2. Run the script: `python3 main.py scrape`

The scraper will take approximately:
- ~2 hours for the whiskies
- ~1 minute for all users
- ~45 minutes for the ratings

The data is saved in a local sqlite database `whisky.db`. To scrape all data again and overwrite the database, run `python3 main.py scrape --reset`.
