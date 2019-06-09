#!/usr/bin/env python3

import datetime
import json
import os
import re
import time

from flask import Flask
from flask import render_template, send_from_directory
from werkzeug.contrib.cache import SimpleCache


print('Setting up CADO Factor')
app = Flask(__name__)

cache = SimpleCache()


def GetData(factor):
    factor_data = cache.get(factor)
    if factor_data is None:
        # TODO handle file not existing
        status_path = os.path.join(app.root_path, factor)
        with open(status_path) as f:
            factor_data = json.load(f)
        cache.set(factor, factor_data, timeout=5 * 60)

    return factor_data


@app.route("/")
def Index():
    number = "2330L.c207"
    data = GetData(number + ".status")
    host_stats, other_stats, random_shuf = data
    max_relations, mtime = other_stats

    RELATION_GOAL = 2.7e9

    newest_eta = random_shuf[-1].split("as ok")[-1].strip(' ()\n')
    last_update = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    updated_delta_s = time.time() - mtime

    found = sum(s[0] for s in host_stats.values())
    relations_done = sum(s[1] for s in host_stats.values())
    all_cpus = sum(s[2] for s in host_stats.values())
    newest_wu = max(s[3] for s in host_stats.values())

    host_hide = ["eifz", "lukerichards", "C5KKONV", "lrichards"]
    for k in list(host_stats.keys()):
        temp = k
        for hide in host_hide:
            temp = temp.replace(hide, '*' * len(hide))
        if temp != k:
            host_stats[temp] = host_stats.pop(k)

    total_line = ("total", (found, relations_done, all_cpus, newest_wu))
    host_stats = [total_line] + sorted(host_stats.items(), key=lambda p: -p[1][1])

    def minimize_line(line):
        line = line.split(" ", 1)[1]
        line = line.replace("Info:", "")
        return line

    random_shuf = list(map(minimize_line, random_shuf))

    return render_template(
        "index.html",
        number=number,
        found=found,
        relations_done=relations_done,
        max_relations=max_relations,
        host_stats=host_stats,
        random_shuf=random_shuf,

        last_update=last_update,
        updated_delta_s=updated_delta_s,
        eta=newest_eta,
    )


@app.route("/favicon.ico")
def Favicon():
    return ""


@app.route('/progress/<name>')
def factor_progress(name):
    return send_from_directory(
        "", name + ".progress.png",
        cache_timeout=120)


if __name__ == "__main__":
    app.run(
        #host='0.0.0.0',
        host = '::',
        port = 5070,
        #debug = False,
        debug = True,
        threaded = True)
