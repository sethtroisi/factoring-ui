#!/usr/bin/env python3
'''
# Look for lines like
PID22558 2019-06-01 22:37:01,063 Info:HTTP server: 123.123.123.123 Sending workunit 2330L.c207_sieving_13849000-13850000 to client <NAME>.<HASH>

# Later you get this line (sadly no mention of work unit here).

PID22558 2019-06-02 14:44:22,699 Debug:HTTP server: 123.123.123.123 "POST /cgi-bin/upload.py HTTP/1.1" 200 -

# And within a ~1 second (upload time)

PID22558 2019-06-02 14:44:23,919 Debug:Lattice Sieving: stderr is: b"# redoing q=14709001, rho=9007715 because 1s buckets are full\n# Fullest level-1s bucket #2094, wrote 3317/3232\n# Average J=22434 for 68 special-q's, max bucket fill -bkmult 1,1s:1.07761\n# Discarded 0 special-q's out of 68 pushed\n# Wasted cpu time due to 1 bkmult adjustments: 48.37\n# Total cpu time 5644.79s [norm 4.91+10.0, sieving 4599.9 (3113.2 + 225.9 + 1260.8), factor 1030.0 (677.1 + 352.9)] (not incl wasted time)\n# Total elapsed time 1502.33s, per special-q 22.0931s, per relation 0.150549s\n# PeakMemusage (MB) = 11333 \n# Total 9979 reports [0.566s/r, 146.8r/sq] in 1.5e+03 elapsed s [375.7% CPU]\n

# After stderr line

PID22558 2019-06-02 14:44:23,919 Debug:Lattice Sieving: Newly arrived stats: {'stats_avg_J': '22434.0 68', 'stats_max_bucket_fill': '1,1s:1.07761', 'stats_total_cpu_time': '5644.79', 'stats_total_time': '1502.33'}

# Next, and Next
PID22558 2019-06-02 14:44:23,920 Debug:Lattice Sieving: Combined stats: {'stats_avg_J': '19767.628076462464 356619', 'stats_max_bucket_fill': '1.0,1s:1.121320', 'stats_total_cpu_time': '30737474.54999989', 'stats_total_time': '8206173.819999958'}
PID22558 2019-06-02 14:44:23,920 Info:Lattice Sieving: Found 9979 relations in '<PATH>/2330L.c207.upload/2330L.c207.14709000-14710000.xbygw_pr.gz', total is now 61549262/2700000000

'''

import argparse
import datetime
import json
import random
import re
import time

from collections import Counter, defaultdict
from operator import itemgetter

import numpy as np

RELATIONS_PTN = re.compile(r"Found ([0-9]*) relations in.*/([0-9_Lc.+]*\.[0-9]{5,12}-[0-9]{5,12})")
STATS_TOTAL_PTN = re.compile(r"'stats_total_cpu_time': '([0-9.]*)',")

# Use 2000-2099 to validate this starts with a date
TOTAL_RELATIONS_PTN = re.compile('(20[12][0-9]-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9:,]*) .* total is now ([0-9]*)')


def get_args():
    parser = argparse.ArgumentParser(
        description="log processor for factoring ui of CADO-NFS",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # TODO find a better way to pass config settings

    # Used to determine <name>.status and <name>.progress.png filenames.
    parser.add_argument('-n', '--name', required=True,
        help="Name of this factoring effort (e.g. 2330L or RSA120)")
    parser.add_argument('--banner', default='',
        help="html banner line to show below ETA status")

    parser.add_argument('-g', '--goal', required=True, type=float,
        help="should match tasks.sieve.rels_wanted")
    parser.add_argument('-l', '--log_path', required=True,
        help="path to log file created by cado-nfs.py")
    parser.add_argument('-s', '--sql_path', required=True,
        help=("path to SQL database created by cado-nfs.py\n"
              "several formats are accepted:\n"
              " * <PATH> (sqlite3 only)\n"
              " * db:sqlite3:<PATH>\n"
              " * db:mysql:<PATH>\n"
              " * db:mysql://<USER>:<PASSWORD>@<HOST>/<DB>\n"
              "NOTE: MySQL is not well tested, please file a bug report\n"
              "if you experience difficulty configuring"))

    parser.add_argument('--client-substitutions', action='append', default=[],
        help="for names in form A:B substitue A => B")
    parser.add_argument('--client-joined', action='append', default=[],
        help="for names in form A:B:C:D join B,C,D under host A")
    parser.add_argument('--host-startswith', default="",
        help="comma seperated list, any client that starts with A is part of host A")

    parser.add_argument('--min-workunits', default=0, type=int,
        help="Minimum workunits to include in host list")

    args = parser.parse_args()
    return args


def parse_log_time(log_time):
    # TODO allow this format to be passed in args.
    return datetime.datetime.strptime(log_time, "%Y-%m-%d %H:%M:%S,%f")


def get_get_host_name(args):
    '''curried version of get_host_name'''
    return lambda client: parse_host_name(args, client)


def parse_host_name(args, client):
    if not client:
        return "<EMPTY>"

    client = re.sub(r'\.[0-9a-f]{7,8}$', '', client)
    client = re.sub(r'\.[0-9]+$', '<X>', client)

    for pairs in args.client_substitutions:
        # NOTE: please don't have clients with : in their names.
        original, replacement = pairs.split(":")
        client = re.sub(original, replacement, client)

    for joins in args.client_joined:
        name, *others = joins.split(":")
        if client.startswith(tuple(others)):
            return name

    if args.host_startswith:
        for host in args.host_startswith.split(","):
            if client.startswith(host):
                return host

    return client


def get_db_data(sql_path):
    SQLITE3_URI = "db:sqlite3:"
    MYSQL_URI = "db:mysql:"

    SQL_QUERY = "select wuid, status, resultclient from workunits"

    if sql_path.startswith(MYSQL_URI):
        import mysql
        import mysql.connector
        import urllib.parse

        uri = sql_path.replace(MYSQL_URI, "mysql:")
        parsed = urllib.parse.urlparse(uri)

        config = {
            'user':     parsed.username,
            'password': parsed.password,
            'host':     parsed.hostname,
            'port':     parsed.port,
            'database': parsed.path.strip('/'),
        }
        config = {k: v for k, v in config.items() if v is not None}
        print("Config:", config)

        db = mysql.connector.connect(**config)
        cur = db.cursor()
        cur.execute(SQL_QUERY)
        results = tuple(cur.fetchall())
        cur.close()
        db.close()
        return results

    import sqlite3
    # Assume local sqlite3 db.
    uri = sql_path.replace(SQLITE3_URI, "")
    print("sqlite3: ", uri)
    with sqlite3.connect(uri) as db:
        cur = db.execute(SQL_QUERY)
        return tuple(cur.fetchall())


def get_client_work(sql_path, get_host_name):
    results = get_db_data(sql_path)
    results = [(wuid.replace("_sieving_", "."), s, v) for wuid, s, v in results]

    # Split workunits into good and bad.
    wuid = {wuid: value for wuid, status, value in results if status == 5}
    bad_wuid = {wuid: value for wuid, status, value in results if status != 5}

    print(f"{len(wuid)} workunits ({len(bad_wuid)} failed), {min(wuid)} to {max(wuid)}")
    client_work = Counter(map(get_host_name, wuid.values()))
    for name, wus in client_work.most_common():
        print("\t{:20} x{} workunits".format(name, wus))
    print()

    clients = sorted(set(wuid.values()))
    hosts = Counter(map(get_host_name, clients))
    print("{} clients, {} hosts".format(len(clients), len(hosts)))
    for name, count in hosts.most_common():
        print("\t{:20} x{} clients (over the run)".format(name, count))
    print()

    return wuid, bad_wuid, client_work


def get_stat_lines(lines):
    stat_lines = []
    for i, line in enumerate(lines):
        match = RELATIONS_PTN.search(line)
        if match:
            wu = match.group(2)
            relations = int(match.group(1))

            # Log seems pretty consistent in this regard, and if stats aren't
            # found a warning is printed which happens 2-30 times in both
            # factoring efforts tested which is managable.
            total_cpu_seconds = 0
            if "Newly arrived stats" in lines[i - 2]:
                cpu_seconds_match = re.search(STATS_TOTAL_PTN, lines[i - 2])
                if cpu_seconds_match:
                    total_cpu_seconds = float(cpu_seconds_match.group(1))
                else:
                    print("Didn't find stats:", lines[i - 2:i])
            else:
                print("Didn't find Newly arrived stats for ", wu)
            log_time = " ".join(line.split(" ")[1:3])

            stat_lines.append((log_time, wu, relations, total_cpu_seconds))
    print()
    return stat_lines


def get_last_log_date(lines):
    for line in reversed(lines):
        parts = line.split()
        if parts[1].startswith('20'):
            return parse_log_time(parts[1] + " " + parts[2])

    print("FALLING BACK TO NOW DATE!")
    return datetime.datetime.uctnow()


def sub_sample(items, max_count):
    """Sample at most max_count elements from items."""
    # random.sample is unhappy with count > len(items)
    count = min(len(items), max_count)
    indexes = sorted(random.sample(range(len(items)), count))
    return [items[i] for i in indexes]


def sub_sample_with_endpoints(items, max_count):
    """sub_sample but include first and last item."""
    assert max_count >= 2, "must be greater than 2 to include endpoints"
    if len(items) <= 2:
        return items
    return [items[0]] + sub_sample(items[1:-1], max_count - 2) + [items[-1]]


def get_stats(wuid, bad_wuid, stat_lines, one_day_ago, client_hosts):
    # wu, relations, cpu_s, last_log, rels_last_24
    client_stats = defaultdict(lambda: [0, 0, 0.0, None, 0])
    host_stats = defaultdict(lambda: [0, 0, 0.0, None, 0])

    # badges
    # first, last
    # min_relations, max_relations
    # min_cpu_seconds, max_cpu_seconds
    client_records = {}

    for log_time, wu, relations, total_cpu_seconds in stat_lines:
        if wu not in wuid:
            if wu not in bad_wuid:
                print("wuid not found", wu, len(wuid))
            continue

        client_name = wuid[wu]
        c_stats = client_stats[client_name]

        host_name = client_hosts[client_name]
        h_stats = host_stats[host_name]

        # Store stats under both client and aggregated host name
        for name, stats in [(client_name, c_stats), (host_name, h_stats)]:
            stats[0] += 1
            stats[1] += relations
            stats[2] += total_cpu_seconds
            stats[3] = max(stats[3], log_time) if stats[3] else log_time
            stats[4] += relations if parse_log_time(log_time) > one_day_ago else 0

            if name not in client_records:
                client_records[name] = [
                    [],
                    log_time, log_time,
                    (relations, wu), (relations, wu),
                    total_cpu_seconds, total_cpu_seconds
                ]
            else:
                record = client_records[name]
                record[1] = min(record[1], log_time) if record[1] else log_time
                record[2] = max(record[2], log_time) if record[2] else log_time

                record[3] = min(record[3], (relations, wu))
                record[4] = max(record[4], (relations, wu))

                record[5] = min(record[5], total_cpu_seconds)
                record[6] = max(record[6], total_cpu_seconds)

    joined_stats = dict(client_stats)
    joined_stats.update(host_stats)

    set_badges(stat_lines, joined_stats, client_records)

    return dict(client_stats), dict(host_stats), client_records


def set_badges(stat_lines, joined_stats, client_records):
    '''Add badges (tuple of name and values) to client_records'''
    wu_relations = sorted(map(itemgetter(2), stat_lines))
    # TODO: This is the only use of numpy it might be worth removing it to
    # avoid the dependency.
    unlucky_relations = np.percentile(wu_relations, 0.1)
    lucky_relations = np.percentile(wu_relations, 99.9)

    print(f"{unlucky_relations:.1f}, {lucky_relations:.1f} [un]lucky relations")
    print()

    for name, record in client_records.items():
        if record[3][0] <= unlucky_relations:
            record[0].append(("unlucky", record[3][0], record[3][1]))

        if record[4][0] >= lucky_relations:
            record[0].append(("lucky", record[4][0], record[4][1]))

        time_delta = parse_log_time(record[2]) - parse_log_time(record[1])
        if time_delta.days > 7:
            weeks = time_delta.total_seconds() / (7 * 24 * 3600)
            record[0].append((
                "weeks",
                int(time_delta.days) // 7,
                "{:.2f} Weeks of workunits".format(weeks),
            ))

        total_cpu_seconds = joined_stats[name][2]

        cpu_year = total_cpu_seconds / (365 * 24 * 3600)
        if cpu_year >= 1:
            record[0].append((
                "CPU-years",
                int(cpu_year),
                "{:.2f} CPU years!".format(cpu_year),
            ))


def relations_last_24hr(log_lines, last_log_date):
    total_found_lines = []
    for line in log_lines:
        match = TOTAL_RELATIONS_PTN.search(line)
        if match:
            datetime_raw, found_raw = match.groups()
            total_found_lines.append((
                parse_log_time(datetime_raw), int(found_raw)
            ))

    total_found_lines.append((
        last_log_date, total_found_lines[-1][1]
    ))

    # Walk forward over all total found lines keeping i pointing to the first
    # line less than 24 hours old.
    total_last_24 = []
    i = j = 0
    while j < len(total_found_lines):
        assert i <= j, (i, j)
        a = total_found_lines[i]
        b = total_found_lines[j]
        time_delta = b[0] - a[0]

        if time_delta.days >= 1:
            # decrease time interval
            i += 1
        else:
            delta = b[1] - a[1]
            total_last_24.append((b[0], delta))
            # increase time interval
            j += 1

    rels_last_24 = total_last_24[-1]
    total_last_24 = sub_sample_with_endpoints(total_last_24, 3000)
    return rels_last_24, total_last_24


def generate_charts(args, eta_lines, total_last_24):
    """
    Generate and save charts:
        Overall progress (out of 100%),
        Relations last 24 hours,
    """
    graph_file = args.name + ".progress.png"
    per_day_graph_file = args.name + ".daily_r.png"

    from matplotlib.dates import DateFormatter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt

    import seaborn as sns
    sns.set()

    log_data = sub_sample_with_endpoints(eta_lines, 2000)
    log_raw_dates = [" ".join(line.split()[1:3]) for line in log_data]
    log_percents = [float(re.search(r"([0-9.]+)%", line).group(1)) for line in log_data]
    log_dates = list(map(parse_log_time, log_raw_dates))

    plt.plot(log_dates, log_percents)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
    ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))

    plt.savefig(graph_file)
    print("Saved as ", graph_file)

    rels_24_dates = [dt for dt, _ in total_last_24]
    rels_24_count = [ct for _, ct in total_last_24]

    ax.clear()

    ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
    ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.01e'))

    plt.plot(rels_24_dates, rels_24_count)
    plt.ylabel("Relations in last 24 hours")

    plt.savefig(per_day_graph_file)
    print("Saved as ", per_day_graph_file)


##### MAIN #####

def main():
    args = get_args()
    get_host_name = get_get_host_name(args)

    wuid, bad_wuid, client_work = get_client_work(args.sql_path, get_host_name)
    client_names = set(wuid.values())
    client_hosts = {client: get_host_name(client) for client in client_names}

    with open(args.log_path) as log_file:
        log_lines = log_file.readlines()

    print()
    print(len(log_lines), "log lines")

    stat_lines = get_stat_lines(log_lines)
    print(len(stat_lines), "Stat lines")
    print()

    last_log_date = get_last_log_date(log_lines)
    print("Last log date:", last_log_date)
    print()

    one_day_ago = last_log_date - datetime.timedelta(days=1)

    ##### Host/Client stats (and badge variables) #####

    client_stats, host_stats, client_records = get_stats(
        wuid, bad_wuid, stat_lines, one_day_ago, client_hosts)

    ##### Print Host Stats #####

    found          = sum(map(itemgetter(0), host_stats.values()))
    relations_done = sum(map(itemgetter(1), host_stats.values()))
    print("Found {} workunits, {} relations ~{:.2f}%".format(
        found, relations_done, 100 * relations_done / args.goal))
    print()

    for host in sorted(host_stats.keys(), key=lambda h: -host_stats[h][2]):
        stat_wu, stat_r, stat_cpus, stat_last, _ = host_stats[host]
        client_record = client_records[host]
        wus = client_work[host]

        percent_total = 100 * stat_r / relations_done
        cpu_days = stat_cpus / (60 * 60 * 24)
        print("\t{:20} x{:5} workunits | stats wu {:5}, relations {:8} ({:4.1f}% total) cpu-days: {:6.1f} last: {}".format(
            host, wus, stat_wu, stat_r, percent_total, cpu_days, stat_last))
        print("\t\t", ", ".join(map(str, client_record[0])))
        print("\t\t", ", ".join(map(str, client_record[1:])))
    print()

    # Verify that we end up with stats on every client that had workunits (from db).
    assert host_stats.keys() == client_work.keys(), (host_stats.keys(), client_work.keys())

    # Remove any client / hosts with less than minimum workunits
    trimmed = 0
    for stats in (client_stats, host_stats):
        for key in list(stats.keys()):
            if stats[key][0] < args.min_workunits:
                trimmed += 1
                stats.pop(key)
    if trimmed > 0:
        print(f"\tRemoved {trimmed} clients/hosts with < {args.min_workunits} workunits")

    ##### Eta logs #####

    eta_lines = [line for line in log_lines if '=> ETA' in line]
    print(f"{len(eta_lines)} ETAs: {eta_lines[-1]}\n")
    eta_logs_sample = sub_sample_with_endpoints(eta_lines, 100)

    ##### Relations per 24 hours #####

    rels_last_24, total_last_24 = relations_last_24hr(log_lines, last_log_date)

    ##### Write status file #####

    status_filename = args.name + ".status"
    with open(status_filename, "w") as status_file:
        json.dump([
            [host_stats, client_stats, client_records, client_hosts],
            [time.time(), args.goal, args.banner],
            eta_logs_sample, rels_last_24[1],
        ], status_file)

    ##### Generate charts #####

    generate_charts(args, eta_lines, total_last_24)


if __name__ == "__main__":
    main()
