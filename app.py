#!/usr/bin/env python3

from datetime import datetime, timedelta
import json
import os
import re
import time

from flask import Flask
from flask import render_template, send_from_directory
from flask_caching import Cache


print('Setting up CADO Factor')
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})


@cache.cached(timeout=5 * 60)
def GetData(factor):
    # TODO handle file not existing
    status_path = os.path.join(app.root_path, factor)
    with open(status_path) as f:
        return json.load(f)

def log_date_str_to_datetime(log_date_str):
    return datetime.strptime(log_date_str, "%Y-%m-%d %H:%M:%S,%f")

@app.route("/")
@app.route("/<number>/")
def Index(number="2330L.c207"):
    data = GetData(number + ".status")
    (host_stats, client_stats, host_records,
     other_stats, random_shuf, rels_last_24) = data
    max_relations, mtime = other_stats

    RELATION_GOAL = 2.7e9

    newest_eta = random_shuf[-1].split("as ok")[-1].strip(' ()\n')
    last_update = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    updated_delta_s = time.time() - mtime
    last_wu = max(host_stat[3] for host_stat in host_stats.values())
    last_wu_time = datetime.timestamp(log_date_str_to_datetime(last_wu))
    wu_delta_s = time.time() - last_wu_time

    workunits_done = sum(s[0] for s in host_stats.values())
    relations_done = sum(s[1] for s in host_stats.values())
    all_cpus = sum(s[2] for s in host_stats.values())
    newest_wu = max(s[3] for s in host_stats.values())

    total_line = ("total", (workunits_done, relations_done, all_cpus, newest_wu, rels_last_24))
    host_stats = [total_line] + sorted(host_stats.items(), key=lambda p: -p[1][1])
    client_stats = sorted(
		client_stats.items(),
		key=lambda p: (p[0].split(".")[0], p[1][3]),
		reverse=True)
    # Filter clients with only one WU.
    client_stats = [(c,v) for c,v in client_stats if v[0] > 1]

    def active_nodes(named_stats):
        return sum(1 for name, stats in named_stats if
            log_date_str_to_datetime(stats[3]) + timedelta(hours=2) > datetime.now())

    active_hosts = active_nodes(host_stats)
    active_clients = active_nodes(client_stats)

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
        goal=RELATION_GOAL,

        workunits_done=workunits_done,
        relations_done=relations_done,
        rels_last_24=rels_last_24,

        max_relations=max_relations,
        host_stats=host_stats,
        active_hosts=active_hosts,
        host_badges=host_badges,
        badge_names=badge_names,

        client_stats=client_stats,
        active_clients=active_clients,

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
