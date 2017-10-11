"""
Quotes API
----------

Download financial quotes via the Google Finance API
"""

import numpy as np

from datetime import datetime

from six.moves.urllib.parse import urlencode
from six.moves.urllib.request import urlopen


def quotes_historical_google(symbol, date1, date2):
    """Get the historical data from Google finance.
    Parameters
    ----------
    symbol : str
        Ticker symbol to query for, for example ``"DELL"``.
    date1 : datetime.datetime
        Start date.
    date2 : datetime.datetime
        End date.
    Returns
    -------
    X : array
        The columns are ``date`` -- datetime, ``open``, ``high``,
        ``low``, ``close`` and ``volume`` of type float.
    """
    params = urlencode({
        'q': symbol,
        'startdate': date1.strftime('%b %d, %Y'),
        'enddate': date2.strftime('%b %d, %Y'),
        'output': 'csv'
    })
    url = 'http://www.google.com/finance/historical?' + params
    with urlopen(url) as response:
        dtype = {
            'names': ['date', 'open', 'high', 'low', 'close', 'volume'],
            'formats': ['object', 'f4', 'f4', 'f4', 'f4', 'f4']
        }
        converters = {0: lambda s: datetime.strptime(s.decode(), '%d-%b-%y')}
        return np.genfromtxt(response, delimiter=',', skip_header=1,
                             dtype=dtype, converters=converters,
                             missing_values='-', filling_values=-1)

