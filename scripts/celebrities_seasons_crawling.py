import json
import re
from itertools import chain

import click
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests


URL = "https://www.spicemarketcolour.com.au"


def fetch_names(url: str) -> list:
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    tags = soup.find_all("img", alt=re.compile(".+"), src=re.compile("https://.+"))
    return [ tag["alt"] for tag in tags]


            
if __name__ == "__main__":
    @click.command()
    @click.option("--dst_path", help="Where to save images", default="data/celebrities.json")
    def run(dst_path: str):
        # fetching season names
        r = requests.get(URL + "/celebrities")
        soup = BeautifulSoup(r.content, "html.parser")
        seasons = [tag["href"] for tag in soup.find_all(href=re.compile(".*(winter)|(spring)|(autumn)|(summer).*"))]

        # crawling
        celebrities = {season.strip("/"): fetch_names(URL + season) for season in tqdm(seasons)}

        # Dump
        with open(dst_path, "w") as fid:
            json.dump(celebrities, fid)
    run()
