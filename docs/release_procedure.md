# Release Procedure

These are the things to do when performing a release.

1. git checkout develop
1. git pull 
    > to ensure in sync with develop latest
1. git flow release start 'new release number'
1. update release number in `leap_ec/__version__.py` (`setup.py` pulls the version number from here)
1. update CHANGELOG.md
1. if necessary sync `docs/source/roadmap.rst` with realistic expectations
1. git flow release finish 'new release number'
1. manually issue the release via github using the release tag and CHANGELOG text
    1. `git push --tags`
1. verify that the release number and final changes have properly propagated to:
    1. github site
       1. Click on 'releases' link on right side of main github page
       2. Click 'Draft a new release'
          1. Fill in tag ID for latest release
          2. Copy CHANGELOG entry
    2. ReadTheDocs `master` and `latest` docs
    3. update pypi.org
       1. https://packaging.python.org/tutorials/packaging-projects/
    4. update roadmap 
1. change to next release + 'dev' to emphasize working on next release on 
   `develop` branch
   1. on `develop`
      1. CHANGELOG.md
      1. Change `leap_ec.__version__` 
