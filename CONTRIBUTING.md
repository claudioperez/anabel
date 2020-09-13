---
title: Contributing
...

Contributions are welcome, and they are greatly appreciated.
**Contents:**

- [Bug reports](#bug-reports)
- [Documentation improvements](#documentation-improvements)
- [Feature requests and feedback](#feature-requests-and-feedback)
- [Development](#development)
  - [Pull Request Guidelines](#pull-request-guidelines)
  - [Tips](#tips)

## Bug reports

When [reporting a bug](https://github.com/claudioperez/anabel/issues) please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

## Documentation improvements

anabel could always use more documentation, whether as part of the
official anabel docs, in docstrings, or even on the web in blog posts, articles, and such.

## Feature requests and feedback

The best way to send feedback is to file an issue at https://github.com/claudioperez/anabel/issues.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that code contributions are welcome :)

## Development

To set up `anabel` for local development:

1. Fork [`anabel`](https://github.com/claudioperez/anabel)
2. Clone your fork locally:

```shell
git clone git@github.com:claudioperez/anabel.git
```

1. Create a branch for local development:

        git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

2. When you're done making changes run all the checks and docs builder with [tox](https://tox.readthedocs.io/en/latest/install.html) one command:

    tox

3. Commit your changes and push your branch to GitHub:

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

4. Submit a pull request through the GitHub website.

### Pull Request Guidelines

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (run `$ tox`) [1]_.
2. Update documentation when there's new API, functionality etc.
3. Add a note to `CHANGELOG.rst` about the changes.
4. Add yourself to `AUTHORS.rst`.

.. [1] If you don't have all the necessary Python versions available locally you can rely on Travis - it will
       [run the tests](https://travis-ci.org/claudioperez/anabel/pull_requests) for each change you add in the pull request.

### Tips

To run a subset of tests::

    tox -e envname -- pytest -k test_myfeature

To run all the test environments in *parallel* (you need to ``pip install detox``)::

    detox
