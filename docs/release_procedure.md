# Release Procedure

These are the things to do when performing a release.

1. git checkout develop
1. git pull 
    > to ensure in sync with develop latest
1. git flow release finish '<new release number>'
1. update release number everywhere
    1. `setup.py`
    1. `docs/source/conf.py`
        > this is for the ReadTheDocs version number
1. update CHANGELOG.md
1. if necessary sync `docs/source/roadmap.rst` with realistic expectations
1. git flow release finish '<new release number>'
1. manually issue the release via github using the release tag and CHANGELOG text
1. verify that the release number and final changes have properly propagated to:
    1. github site
    1. ReadTheDocs `master` and `latest` docs
    1. update pypi.org
