CREATE TABLE IF NOT EXISTS distillery (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NULL,
    url TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS category (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bottler (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tag (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasting_note (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    country TEXT NULL
);

CREATE TABLE IF NOT EXISTS whisky (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    bottling_serie TEXT NULL,
    category_id INTEGER NULL,
    bottler_id INTEGER NULL,
    age INTEGER NULL,
    bottled_year INTEGER NULL,
    strength REAL NULL,
    size INTEGER NULL,
    url TEXT NOT NULL,
    price INTEGER NULL,
    availability INTEGER NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES category (id),
    FOREIGN KEY (bottler_id) REFERENCES bottler (id)
);

CREATE TABLE whisky_distillery (
    whisky_id INTEGER NOT NULL,
    distillery_id INTEGER NOT NULL,
    FOREIGN KEY (whisky_id) REFERENCES whisky (id),
    FOREIGN KEY (distillery_id) REFERENCES distillery (id)
);

CREATE TABLE IF NOT EXISTS whisky_tags (
    whisky_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    FOREIGN KEY (whisky_id) REFERENCES whisky (id),
    FOREIGN KEY (tag_id) REFERENCES tag (id)
);

CREATE TABLE IF NOT EXISTS whisky_tasting_note (
    whisky_id INTEGER NOT NULL,
    tasting_note_id INTEGER NOT NULL,
    votes INTEGER NULL,
    FOREIGN KEY (whisky_id) REFERENCES whisky (id),
    FOREIGN KEY (tasting_note_id) REFERENCES tasting_note (id)
);

CREATE TABLE IF NOT EXISTS whisky_rating (
    username TEXT NOT NULL,
    whisky_id INTEGER NOT NULL,
    rating INTEGER NULL,
    FOREIGN KEY (username) REFERENCES user (username),
    FOREIGN KEY (whisky_id) REFERENCES whisky (id)
);
