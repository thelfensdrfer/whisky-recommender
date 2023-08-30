from whisky.scrape import download, insert_whisky


def test_cao_ila_12():
    URL = "https://www.whiskybase.com/whiskies/whisky/23/caol-ila-12-year-old#whisky-note-holder"

    response = insert_whisky("Caol Ila", download(URL).text)
    assert type(response) == dict
