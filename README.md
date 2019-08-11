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

### Example usage

```bash
./log_processor.py -n 2330L.c207 -g 3e9 --output 2330L.c207 -s 2330L.c207.db --l 2330L.c207.log
```

```bash
$ crontab -l
*/15      * * * * /var/www/factoring/update.sh > /tmp/factoring.log 2>&1

$ cat /var/www/factoring/update.sh
date
cd "<PATH>"

number="2330L.c207"

sshpass -p "<PASSWORD>" rsync -v "<USER>@<REMOTE_HOST>:<PATH>/${number}.{db,log}" .

./log_processor.py -n "$number" -g 3e9 --output "$number" -s "$number.db" --l "$number.log"
```


### TODO cleanups

* [1/2] Remove constants from logs_processor.py
  * [x] Config via argparse
  * [ ] Config for host regexes...
* [ ] Test with local run
  * [ ] Add command example(s)
  * [x] Add crontab example(s)
* [ ] Remove numpy.percentile

### Future Features

* [ ] Support MySQL (and db uri)
* [ ] Display information about
  * [ ] Paramenters (minus paths)
  * [ ] Failed / timedout tasks
  * [ ] Polyselect