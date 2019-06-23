#!/usr/bin/env python3

from datetime import datetime
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
    (host_stats, client_stats, host_records,
     other_stats, random_shuf, rels_last_24) = data
    max_relations, mtime = other_stats

    RELATION_GOAL = 2.7e9


    newest_eta = random_shuf[-1].split("as ok")[-1].strip(' ()\n')
    last_update = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    updated_delta_s = time.time() - mtime
    last_wu = max(host_stat[3] for host_stat in host_stats.values())
    last_wu_time = datetime.timestamp(datetime.strptime(last_wu, "%Y-%m-%d %H:%M:%S,%f"))
    wu_delta_s = time.time() - last_wu_time

    found = sum(s[0] for s in host_stats.values())
    relations_done = sum(s[1] for s in host_stats.values())
    all_cpus = sum(s[2] for s in host_stats.values())
    newest_wu = max(s[3] for s in host_stats.values())

    total_line = ("total", (found, relations_done, all_cpus, newest_wu))
    host_stats = [total_line] + sorted(host_stats.items(), key=lambda p: -p[1][1])
    client_stats = sorted(client_stats.items())

    def minimize_line(line):
        line = line.split(" ", 1)[1]
        line = line.replace("Info:", "")
        return line

    random_shuf = list(map(minimize_line, random_shuf))

    host_badges = {host: records[0] for host, records in host_records.items()}
    badges = set(badge[0] for badges in host_badges.values() for badge in badges)
    badge_names = {
        "unlucky": "badge-danger",
        "lucky":   "badge-success",
        "CPU-years": "badge-secondary",
        "weeks":   "badge-dark",
    }
    if badges > badge_names.keys():
        print ("No badge type for:", badges - badge_names.keys())

    return render_template(
        "index.html",
        number=number,
        found=found,
        relations_done=relations_done,
        rels_last_24=rels_last_24,

        max_relations=max_relations,
        host_stats=host_stats,
        host_badges=host_badges,
        badge_names=badge_names,

        client_stats=client_stats,

        random_shuf=random_shuf,

        eta=newest_eta,
        last_update=last_update,
        updated_delta_s=updated_delta_s,
        last_wu=last_wu,
        wu_delta_s=wu_delta_s,


    )


@app.route("/favicon.ico")
def Favicon():
    return ""


@app.route('/progress/<name>/<graph>')
def factor_progress(name, graph):
    print (graph)
    return send_from_directory(
        app.root_path, name + "." + graph + ".png",
        cache_timeout=120)


if __name__ == "__main__":
    app.run(
        #host='0.0.0.0',
        host = '::',
        port = 5070,
        #debug = False,
        debug = True,
        threaded = True)
