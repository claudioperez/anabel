
from IPython.display import display_html

def disp_sbs(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style = "display:inline"'), raw=True)