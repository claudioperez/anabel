"""
Sphinx Bootstrap theme.

Adapted for the pandas documentation.
"""
import os

import sphinx.builders.html
from sphinx.errors import ExtensionError

from .bootstrap_html_translator import BootstrapHTML5Translator
import docutils

__version__ = "0.0.1.dev0"


# -----------------------------------------------------------------------------
# Sphinx monkeypatch for adding toctree objects into context


def convert_docutils_node(list_item, only_pages=False):
    if not list_item.children:
        return None
    reference = list_item.children[0].children[0]
    title = reference.astext()
    url = reference.attributes["refuri"]
    active = "current" in list_item.attributes["classes"]

    if only_pages and '#' in url:
        return None

    nav = {}
    nav["title"] = title
    nav["url"] = url
    nav["children"] = []
    nav["active"] = active

    if len(list_item.children) > 1:
        for child_item in list_item.children[1].children:
            child_nav = convert_docutils_node(child_item, only_pages=only_pages)
            if child_nav is not None:
                nav["children"].append(child_nav)

    return nav


def update_page_context(self, pagename, templatename, ctx, event_arg):
    from sphinx.environment.adapters.toctree import TocTree

    def get_nav_object(**kwds):
        """Return a list of nav links that can be accessed from Jinja."""
        toctree = TocTree(self.env).get_toctree_for(
            pagename, self, collapse=True, **kwds
        )

        # Grab all TOC links from any toctrees on the page
        toc_items = [item for child in toctree.children for item in child
                     if isinstance(item, docutils.nodes.list_item)]

        nav = []
        for child in toc_items:
            child_nav = convert_docutils_node(child, only_pages=True)
            nav.append(child_nav)

        return nav

    def get_page_toc_object():
        """Return a list of within-page TOC links that can be accessed from Jinja."""
        self_toc = TocTree(self.env).get_toc_for(pagename, self)

        try:
            nav = convert_docutils_node(self_toc.children[0])
            return nav
        except:
            return {}

    ctx["get_nav_object"] = get_nav_object
    ctx["get_page_toc_object"] = get_page_toc_object
    return None


sphinx.builders.html.StandaloneHTMLBuilder.update_page_context = update_page_context

# -----------------------------------------------------------------------------

def setup_edit_url(app, pagename, templatename, context, doctree):
    """Add a function that jinja can access for returning the edit URL of a page."""
    def get_edit_url():
        """Return a URL for an "edit this page" link."""
        required_values = ["github_user", "github_repo", "github_version"]
        for val in required_values:
            if not context.get(val):
                raise ExtensionError("Missing required value for `edit this page` button. "
                                        "Add %s to your `html_context` configuration" % val)

        github_user = context['github_user']
        github_repo = context['github_repo']
        github_version = context['github_version']
        file_name = f"{pagename}{context['page_source_suffix']}"

        # Make sure that doc_path has a path separator only if it exists (to avoid //)
        doc_path = context.get("doc_path", "")
        if doc_path and not doc_path.endswith("/"):
            doc_path = f"{doc_path}/"

        # Build the URL for "edit this button"
        url_edit = f"https://github.com/{github_user}/{github_repo}/edit/{github_version}/{doc_path}{file_name}"
        return url_edit

    context['get_edit_url'] = get_edit_url


# -----------------------------------------------------------------------------


def get_html_theme_path():
    """Return list of HTML theme paths."""
    theme_path = os.path.abspath(os.path.dirname(__file__))
    return [theme_path]


def setup(app):
    theme_path = get_html_theme_path()[0]
    app.add_html_theme("pandas_sphinx_theme", theme_path)
    app.set_translator("html", BootstrapHTML5Translator)
    app.connect("html-page-context", setup_edit_url)
