import requests
import datetime
from typing import Union, Tuple
from functools import lru_cache
import pandas as pd
import numpy as np


@lru_cache
def get_sunrise_sunset(lon: float, lat: float, date: Union[str, datetime.datetime]) -> Tuple[
    datetime.datetime, datetime.datetime]:
    """Get sunrise and sunset times for a given date and location in UTC."""
    if isinstance(date, (datetime.datetime, datetime.date)):
        date = date.strftime("%Y-%m-%d")

    r = requests.get(
        "https://api.sunrise-sunset.org/json",
        params={
            "lat": lat,
            "lng": lon,
            "date": date,
            "formatted": 0  # get ISO formatted dates
        }
    )
    # Times in UTC!
    sunrise_str = r.json()["results"]["sunrise"]
    sunset_str = r.json()["results"]["sunset"]
    return datetime.datetime.fromisoformat(sunrise_str), datetime.datetime.fromisoformat(sunset_str)


def get_daytime_mask(datetime_index_utc: pd.DatetimeIndex, lon: float, lat: float) -> np.ndarray:
    """Get a mask for a given datetime index where True represents daytime."""
    daytime = np.zeros(len(datetime_index_utc), dtype=bool)

    # Loop over unique days
    unique_days = datetime_index_utc.normalize().unique()
    for day in unique_days:
        # Get sunrise and sunset times for the current day and remove timezone info
        sunrise, sunset = get_sunrise_sunset(lon=lon, lat=lat, date=day)
        sunrise = sunrise.replace(tzinfo=None)
        sunset = sunset.replace(tzinfo=None)

        # Create a mask for the current day where True represents daytime
        day_mask_i = (datetime_index_utc >= sunrise) & (datetime_index_utc <= sunset)
        daytime[day_mask_i] = True

    return daytime


if __name__ == '__main__':
    sr, ss = get_sunrise_sunset(10, 52, "2023-08-01")
    print(sr, ss)
