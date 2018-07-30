# Copyright (C) Microsoft Corporation.  All rights reserved.
"""Prepare the data for pairing duplicate and original questions."""

import pandas as pd
import re
import gzip
import requests


def read_csv_gz(url, **kwargs):
    """Load raw data from a .tsv.gz file into Pandas data frame."""
    df = pd.read_csv(gzip.open(requests.get(url, stream=True).raw, mode='rb'),
                     sep='\t', encoding='utf8', **kwargs)
    return df


def clean_text(text):
    """Remove embedded code chunks, links/URLs, and HTML tags."""
    if not isinstance(text, str):
        return text
    text = re.sub('<pre><code>.*?</code></pre>', '', text)
    text = re.sub('<a[^>]+>(.*)</a>', replace_link, text)
    return re.sub('<[^>]+>', '', text)


def replace_link(match):
    if re.match('[a-z]+://', match.group(1)):
        return ''
    else:
        return match.group(1)
