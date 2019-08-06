Factoring UI for cado-nfs
=========================

This file describes how to use logs_processor and flask to run a website showing
sieving progress.

See [factoring.cloudygo.com](http://factoring.cloudygo.com) for an example of
what this can look like.

### Python 3

Python 3, version 3.7 or greater, is suggested. Older versions of Python 3 may
work but have not been tested.

### Packages

Some extra packages are required (Flask, Flask-Cache, matplotlib, seaborn) see
requirements.txt for a full list.

These can be installed with

```bash
pip install -r requirements.txt
```

### TODO cleanups

* [ ] Remove constants from logs_processor.py
  * [ ] Config for rel_goal, name, host regexes...
* [ ] Read SQL db from cmdline
* [ ] Test with local run
  * [ ] Add command example(s)
  * [ ] Add crontab example(s)
* [ ] Remove numpy.percentile

### Future Features

* [ ] Support MySQL
* [ ] Display information about
  * [ ] Paramenters (minus paths)
  * [ ] Failed / timedout tasks
  * [ ] Polyselect
