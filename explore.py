# Copyright (C) Microsoft Corporation.  All rights reserved.
"""Create exploration files."""

import os


def to_html(path, df, column, id='Id'):
    with open(os.path.join(path, column + '.html'), 'w+') as file:
        for _, Id, Text in df[[id, column]].itertuples():
            print("<h1>{}</h1>\n{}\n\n".format(Id, Text), file=file)
