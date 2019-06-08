#!/usr/bin/env python3

import datetime
import json
import os
import re
import time

from flask import Flask
from flask import render_template
from werkzeug.contrib.cache import SimpleCache


print('Setting up CADO Factor')
app = Flask(__name__)

cache = SimpleCache()


def GetData(factor):
    factor_data = cache.get(factor)
    if factor_data is None:
        # TODO handle file not existing
        status_path = os.path.join(app.root_path, factor)
        mtime = os.path.getmtime(status_path)

        with open(status_path) as f:
            factor_data = json.load(f) + [mtime]
        cache.set(factor, factor_data, timeout=5 * 60)

    return factor_data


@app.route("/")
def Index():
    factor = "2330L.c207"
    data = GetData(factor + ".status")
    host_stats, random_shuf, mtime = data

    RELATION_GOAL = 2.7e9

    newest_eta = random_shuf[-1].split("as ok")[-1].strip(' ()\n')
    last_update = datetime.datetime.fromtimestamp(mtime).isoformat().replace('T', ' ')
    hours_ago = (time.time() - mtime) / 3600

    found = sum(s[0] for s in host_stats.values())
    relations_done = sum(s[1] for s in host_stats.values())
    all_cpus = sum(s[2] for s in host_stats.values())
    newest_wu = max(s[3] for s in host_stats.values())

    host_hide = ["birch4", "buster", "eifz", "lukerichards", "C5KKONV", "lrichards"]
    for k in list(host_stats.keys()):
        temp = k
        for hide in host_hide:
            temp = temp.replace(hide, '*' * len(hide))
        if temp != k:
            host_stats[temp] = host_stats.pop(k)

    total_line = ("total", (found, relations_done, all_cpus, newest_wu))
    host_stats = [total_line] + sorted(host_stats.items(), key=lambda p: -p[1][2])


    return render_template(
        "index.html",
        found=found,
        relations_done=relations_done,
        host_stats=host_stats,
        random_shuf=random_shuf,

        last_update=last_update,
        hours_ago=hours_ago,
        eta=newest_eta,
    )


@app.route("/favicon.ico/")
def Favicon():
    return ""


if __name__ == "__main__":
    app.run(
        #host='0.0.0.0',
        host = '::',
        port = 5070,
        #debug = False,
        debug = True,
        threaded = True)
