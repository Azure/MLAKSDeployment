import pandas as pd
import re
import math
import gzip
import requests
import json
import ipywidgets as widgets


def read_csv_gz(url, **kwargs):
    """Load raw data from a .tsv.gz file into Pandas data frame."""
    df = pd.read_csv(gzip.open(requests.get(url, stream=True).raw, mode='rb'),
                     sep='\t', encoding='utf8', **kwargs)
    return df.set_index('Id')


def clean_text(text):
    """Remove embedded code chunks, HTML tags and links/URLs."""
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


def round_sample(X, frac=0.1, min=1):
    """Sample X ensuring at least min samples are selected."""
    n = max(min, math.floor(len(X) * frac))
    return X.sample(n)


def round_sample_strat(X, strat, **kwargs):
    """Sample X ensuring at least min samples are selected."""
    return X.groupby(strat).apply(round_sample, **kwargs)


def random_merge(A, B, N=20, on='AnswerId', key='key', n='n'):
    """Pair all rows of A with 1 matching row on "on" and N-1 random rows from B
    """
    assert key not in A and key not in B
    X = A.copy()
    X[key] = A[on]
    Y = B.copy()
    Y[key] = B[on]
    match = X.merge(Y, on=key).drop(key, axis=1)
    match[n] = 0
    df_list = [match]
    for i in A.index:
        X = A.loc[[i]]
        Y = B[B[on] != X[on].iloc[0]].sample(N-1)
        X[key] = 1
        Y[key] = 1
        Z = X.merge(Y, how='outer', on=key).drop(key, axis=1)
        Z[n] = range(1, N)
        df_list.append(Z)
    df = pd.concat(df_list, ignore_index=True)
    return df


def text_to_json(text):
    return json.dumps({'input': '{0}'.format(text)})


def write_json_to_file(json_dict, filename, mode='w'):
    with open(filename, mode) as outfile:
        json.dump(json_dict, outfile, indent=4, sort_keys=True)
        outfile.write('\n\n')


def buttons_and_texts(data, id, answerid, text, handle_click,
                      layout=widgets.Layout(width="100%"), n=15):
    """Construct buttons, text areas, and a mapping from IDs to text areas."""
    items = []
    text_map = {}
    for i in range(min(n, len(data))):
        button = widgets.Button(description=data.iloc[i][id])
        button.answerid = data.iloc[i][answerid] if answerid in data else None
        button.open = False
        button.on_click(handle_click)
        items.append(button)
        text_area = widgets.Textarea(data.iloc[i][text],
                                     placeholder=data.iloc[i][id],
                                     layout=layout)
        items.append(text_area)
        text_map[data.iloc[i][id]] = text_area
    return items, text_map


def duplicates_questions_widget(duplicates, questions,
                                layout=widgets.Layout(width="100%")):
    """Construct a duplicates and questions exploration widget."""
    global duplicates_id, duplicates_answerid, duplicates_text
    global duplicates_title, questions_id, questions_answerid, questions_text
    global questions_title
    # Construct the duplicates Tab of buttons and text areas.
    duplicates_items, duplicates_map = buttons_and_texts(
        duplicates, duplicates_id, duplicates_answerid, duplicates_text,
        duplicates_click, n=duplicates.shape[0])
    duplicates_tab = widgets.Tab(
        [widgets.VBox(duplicates_items, layout=layout)],
        layout=widgets.Layout(width="100%", height="500px", overflow_y="auto"))
    duplicates_tab.set_title(0, duplicates_title)
    # Construct the questions Tab of buttons and text areas.
    questions_items, questions_map = buttons_and_texts(
        questions, questions_id, questions_answerid, questions_text,
        questions_click, n=questions.shape[0])
    questions_tab = widgets.Tab(
        [widgets.VBox(questions_items, layout=layout)],
        layout=widgets.Layout(width="100%", height="500px", overflow_y="auto"))
    questions_tab.set_title(0, questions_title)
    # Put both tabs in an HBox.
    duplicates_questions = widgets.HBox([duplicates_tab, questions_tab],
                                        layout=layout)
    return duplicates_map, questions_map, duplicates_questions


def questions_click(button):
    """Respond to a click on a question button."""
    global questions_map
    if button.open:
        questions_map[button.description].rows = None
        button.open = False
    else:
        questions_map[button.description].rows = 10
        button.open = True


def duplicates_click(button):
    """Respond to a click on a duplicate button."""
    global duplicates_map
    if select_duplicate(button):
        duplicates_map[button.description].rows = 10
        button.open = True
    else:
        if button.open:
            duplicates_map[button.description].rows = None
            button.open = False
        else:
            duplicates_map[button.description].rows = 10
            button.open = True


def select_duplicate(button):
    """Update the displayed questions to correspond to the button's duplicate
    selections. Returns whether or not the selected duplicate changed.
    """
    global selected_button, questions_map, duplicates_questions, score_text
    global questions, questions_id, questions_answerid, questions_text
    global questions_display, questions_button_color, questions_button_score
    if 'selected_button' not in globals() or button != selected_button:
        if 'selected_button' in globals():
            selected_button.style.button_color = None
            selected_button.style.font_weight = ''
        selected_button = button
        selected_button.style.button_color = 'yellow'
        selected_button.style.font_weight = 'bold'
        duplicates_text = duplicates_map[selected_button.description].value
        questions_scores = score_text(duplicates_text)
        ordered_questions = questions.loc[questions_scores[questions_id]]
        questions_items, questions_map = buttons_and_texts(
            ordered_questions, questions_id, questions_answerid,
            questions_text, questions_click, n=questions_display)
        if (questions_button_color is True
            and selected_button.answerid is not None):
            set_button_color(questions_items[::2], selected_button.answerid)
        if questions_button_score is True:
            questions_items = [
                item
                for button, text_area in zip(*[iter(questions_items)]*2)
                for item in (add_button_prob(button, questions_scores),
                             text_area)]
        duplicates_questions.children[1].children[0].children = questions_items
        duplicates_questions.children[1].set_title(
            0, selected_button.description)
        return True
    else:
        return False


def add_button_prob(button, questions_scores):
    """Return an HBox containing button and its probability."""
    global score_label, score_scale, questions_probability
    id = button.description
    prob = widgets.Label(score_label
                         + ': '
                         + str(int(math.ceil(
                             score_scale
                             * questions_scores.loc[id][
                                 questions_probability]))))
    return widgets.HBox([button, prob])


def set_button_color(button, answerid):
    """Set each button's color according to its label."""
    for i in range(len(button)):
        button[i].style.button_color = (
            'lightgreen' if button[i].answerid == answerid else None)


def read_questions(path, id, answerid):
    """Read in a questions file with at least Id and AnswerId columns."""
    questions = pd.read_csv(path, sep='\t', encoding='latin1')
    questions[id] = questions[id].astype(str)
    questions[answerid] = questions[answerid].astype(str)
    questions = questions.set_index(id, drop=False)
    questions.sort_index(inplace=True)
    return questions
