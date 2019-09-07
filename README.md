Factoring UI for cado-nfs
=========================

This file describes how to use log_processor.py and flask to run a website
showing sieving progress.

See [factoring.cloudygo.com](http://factoring.cloudygo.com) for an example of
what this can look like.

See [Log Processing](#log-processing) and [Frontend](#frontend) on how to get
started

### Prerequisites

##### Python 3
Python 3, version 3.7 or greater, is suggested. Older versions of Python 3 may
work but have not been tested.

##### Packages

Some extra packages are required (Flask, Flask-Cache, matplotlib, seaborn) see
[requirements.txt](requirements.txt) for a full list.

These can be installed with

```bash
pip install -r requirements.txt
```

### Log Processing

The first step is to anaylze the cado-nfs database and logs.

`log_processor.py` needs access to the log file (via `-l` or `--log_path`)
and the database file (via `-s` or `--sql_path`) see `log_processor.py -h`
for help on understanding options and `--sql_path` format.

```bash
./log_processor.py -n 2330L.c207 -g 3e9 -s 2330L.c207.db -l 2330L.c207.log
./log_processor.py -n X -g 3e6 -s "db:mysql://testuser:testpasswd@localhost/db_name" -l X.log

```

This can be automated with cron

```bash
$ crontab -l
*/15      * * * * /var/www/factoring/update.sh > /tmp/factoring.log 2>&1

$ cat /var/www/factoring/update.sh
date
cd "<PATH>"

number="2330L.c207"

# If files aren't on the same computer
sshpass -p "<PASSWORD>" rsync -v "<USER>@<REMOTE_HOST>:<PATH>/${number}.{db,log}" .

./log_processor.py -n "$number" -g 3e9 --output "$number" -s "$number.db" -l "$number.log"
```

# Frontend

The Flask build-in server is great to get started.
Later when you want to deploy in production check out Flask's documentation
on [Deployment Options](https://flask.palletsprojects.com/en/1.1.x/deploying/)

```bash
FLASK_APP=app.py flask run --port 5100

# To expose the site on more than localhost
FLASK_APP=app.py flask run --port 5100 --host 0.0.0.0

# To run in debug mode for local developement
FLASK_DEBUG=1 FLASK_APP=app.py flask run --port 5100
```

Congrats! the [factoring-ui](screenshots/main-page.png) should be running on
[localhost:5100](http://localhost:5100)


### Future Features

* [x] Support MySQL (and db uri)
* [ ] Display information about
  * [ ] Paramenters (minus paths)
  * [ ] Failed / timedout tasks
  * [ ] Polyselect
* P1
  * [ ] Pass an arg for name (e.g. 13_945, 2330L.c207) to app.py
  * [ ] Gracefully handle case where log is empty.
* P2
  * [ ] Read relation goal from sql.
  * [ ] Support config from file in parameter format
  * [ ] Verify all logs start with same dateformat (or pass as arg)
  * [ ] Remove numpy.percentile
