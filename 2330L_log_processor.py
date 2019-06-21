'''
# Look for lines like
PID22558 2019-06-01 22:37:01,063 Info:HTTP server: 123.123.123.123 Sending workunit 2330L.c207_sieving_13849000-13850000 to client <NAME>.<HASH>

# Later you get this line (sadly no mention of work unit here.
PID22558 2019-06-02 14:44:22,699 Debug:HTTP server: 123.123.123.123 "POST /cgi-bin/upload.py HTTP/1.1" 200 -
# And within a ~1 second (upload time)
PID22558 2019-06-02 14:44:23,919 Debug:Lattice Sieving: stderr is: b"# redoing q=14709001, rho=9007715 because 1s buckets are full\n# Fullest level-1s bucket #2094, wrote 3317/3232\n# Average J=22434 for 68 special-q's, max bucket fill -bkmult 1,1s:1.07761\n# Discarded 0 special-q's out of 68 pushed\n# Wasted cpu time due to 1 bkmult adjustments: 48.37\n# Total cpu time 5644.79s [norm 4.91+10.0, sieving 4599.9 (3113.2 + 225.9 + 1260.8), factor 1030.0 (677.1 + 352.9)] (not incl wasted time)\n# Total elapsed time 1502.33s, per special-q 22.0931s, per relation 0.150549s\n# PeakMemusage (MB) = 11333 \n# Total 9979 reports [0.566s/r, 146.8r/sq] in 1.5e+03 elapsed s [375.7% CPU]\n"
# After stderr line
PID22558 2019-06-02 14:44:23,919 Debug:Lattice Sieving: Newly arrived stats: {'stats_avg_J': '22434.0 68', 'stats_max_bucket_fill': '1,1s:1.07761', 'stats_total_cpu_time': '5644.79', 'stats_total_time': '1502.33'}
# Next, and Next
PID22558 2019-06-02 14:44:23,920 Debug:Lattice Sieving: Combined stats: {'stats_avg_J': '19767.628076462464 356619', 'stats_max_bucket_fill': '1.0,1s:1.121320', 'stats_total_cpu_time': '30737474.54999989', 'stats_total_time': '8206173.819999958'}
PID22558 2019-06-02 14:44:23,920 Info:Lattice Sieving: Found 9979 relations in '/home/vbcurtis/ssd/cado-nfs/2330Ljob/2330L.c207.upload/2330L.c207.14709000-14710000.xbygw_pr.gz', total is now 61549262/2700000000

'''

import datetime
import json
import random
import re
import sqlite3
import time

from collections import Counter, defaultdict
from operator import itemgetter

import numpy as np

RELATIONS_GOAL = 2.7e9

RELATIONS_PTN = re.compile("Found ([0-9]*) relations in.*(2330L\.c207\.[0-9]*-[0-9]*)")

NUMBER_NAME = "2330L.c207"

SQL_FILE = NUMBER_NAME + ".db"
LOG_FILE = NUMBER_NAME + ".log"

STATUS_FILE = NUMBER_NAME + ".status"
GRAPH_FILE  = NUMBER_NAME + ".progress.png"


def parse_log_time(log_time):
    return datetime.datetime.strptime(log_time, "%Y-%m-%d %H:%M:%S,%f")


def host_name(client):
    if not client:
        return" <EMPTY>"
    client = re.sub('vebis.*', 'vebis<X>', client)
    client = re.sub('lukerichards-.*', 'lukerichards-<COMP>', client)
    return client.split(".")[0]

##### MAIN #####

with sqlite3.connect(SQL_FILE) as db:
    # About 40 workunits have status = 7????
    cur = db.execute("select wuid, resultclient from workunits")
    wuid = dict(cur.fetchall())
wuid = {key.replace("_sieving_", "."): value for key, value in wuid.items()}


print()
print ("{} workloads, {} to {}".format(
    len(wuid),
    min(wuid.keys()),
    max(wuid.keys())))
client_work = Counter(map(host_name, wuid.values()))
for host, wus in client_work.most_common():
    print ("\t{:20} x{} workuints".format(host, wus))
print ()


clients = sorted(set(v for v in wuid.values() if v is not None))
hosts = Counter(map(host_name, clients))
print ("{} clients, {} hosts".format(len(clients), len(hosts)))
for host, count in hosts.most_common():
    print ("\t{:20} x{} clients (over the run)".format(host, count))
print ()

with open(LOG_FILE) as f:
    lines = f.readlines()

print()
print(len(lines), "log lines")

stat_lines = []
for i, line in enumerate(lines):
    match = RELATIONS_PTN.search(line)
    if match:
        wu = match.group(2)
        relations = int(match.group(1))
        #print (wu, relations, "\t", lines[i-2])

        total_cpu_seconds = 0
        if "Newly arrived stats" in lines[i-2]:
            cpu_seconds_match = re.search(r"'stats_total_cpu_time': '([0-9.]*)',", lines[i-2])
            if cpu_seconds_match:
                total_cpu_seconds = float(cpu_seconds_match.group(1))
            else:
                print ("Didn't find stats:", lines[i-2:i])
        else:
            print ("Didn't find Newly arrived stats?")
        log_time = " ".join(line.split(" ")[1:3])

        stat_lines.append((log_time, wu, relations, total_cpu_seconds))
print ()


##### Varibles for badges #####

# wu, relations, cpu_s, last, is
host_stats = defaultdict(lambda: [0, 0, 0.0, None])

# badges
# first, last
# min_relations, max_relations
# min_cpu_seconds, max_cpu_seconds
host_records = defaultdict(lambda: [[], None, None, (10**6, ""), (0, ""), 10**6, 0])

for log_time, wu, relations, total_cpu_seconds in stat_lines:
    host_name_full = wuid[wu]
    host_name_short = host_name(wuid[wu])

    # Short stats under both short and full name
    for host in set([host_name_full, host_name_short]):
        host_stat = host_stats[host]
        host_stat[0] += 1
        host_stat[1] += relations
        host_stat[2] += total_cpu_seconds
        host_stat[3] = max(host_stat[3], log_time) if host_stat[3] else log_time

        host_record = host_records[host]
        host_record[1] = min(host_record[1], log_time) if host_record[1] else log_time
        host_record[2] = max(host_record[2], log_time) if host_record[2] else log_time

        host_record[3] = min(host_record[3], (relations, wu))
        host_record[4] = max(host_record[4], (relations, wu))

        host_record[5] = min(host_record[5], total_cpu_seconds)
        host_record[6] = max(host_record[6], total_cpu_seconds)


wu_relations = sorted(map(itemgetter(2), stat_lines))
max_relations = wu_relations[-1]
unlucky_relations = np.percentile(wu_relations, 1)
lucky_relations = np.percentile(wu_relations, 99)

print (unlucky_relations, lucky_relations, "[un]lucky relations")

for host, host_record in host_records.items():
    if host_record[3][0] <= unlucky_relations:
        host_record[0].append((
            "unlucky",
            host_record[3][0],
            host_record[3][1],
        ))

    if host_record[4][0] >= lucky_relations:
        host_record[0].append((
            "lucky",
            host_record[4][0],
            host_record[4][1],
        ))

    time_delta = parse_log_time(host_record[2]) - parse_log_time(host_record[1])
    if time_delta.days > 7:
        weeks = time_delta.total_seconds() / (7 * 24 * 3600)
        host_record[0].append((
            "weeks",
            int(time_delta.days) // 7,
            "{:.2f} Weeks of workunits".format(weeks),
        ))

    total_cpu_seconds = host_stats[host][2]
    cpu_year = total_cpu_seconds / (365 * 24 * 3600)
    if cpu_year >= 2:
        host_record[0].append((
            "CPU-years",
            int(cpu_year),
            "{:.2f} CPU years!".format(cpu_year),
        ))

##### Output #####

client_stats = {h: v for h, v in host_stats.items() if host_name(h) != h}
for h in client_stats:
    host_stats.pop(h)

found          = sum(map(itemgetter(0), host_stats.values()))
relations_done = sum(map(itemgetter(0), host_stats.values()))
print ("Found {} workunits, {} relations ~{:.2f}%".format(
    found, relations_done, 100 * relations_done / RELATIONS_GOAL))
print ()

for host in sorted(host_stats.keys(), key=lambda h: -host_stats[h][2]):
    stat_wu, stat_r, stat_cpus, stat_last = host_stats[host]
    host_record = host_records[host]
    wus = client_work[host]
    print ("\t{:20} x{:5} workunits | stats wu {:5}, relations {:8} ({:4.1f}% total) cpu-days: {:6.1f} last: {}".format(
        host, wus, stat_wu, stat_r, 100 * stat_r / relations_done, stat_cpus / 3600 / 24, stat_last))
    print ("\t\t", ", ".join(map(str, host_record[0])))
    print ("\t\t", ", ".join(map(str, host_record[1:])))
print ()

assert host_stats.keys() == client_work.keys(), host_stats.keys()

eta_lines = [line for line in lines if '=> ETA' in line]
print (f"{len(eta_lines)} ETAs: {eta_lines[-1]}")
print ()

random_shuf = [l for i, l in sorted(random.sample(list(enumerate(eta_lines)), 100))] + eta_lines[-1:]
with open(STATUS_FILE, "w") as f:
    json.dump([host_stats, client_stats, host_records, [max_relations, time.time()], random_shuf], f)


#--------------------------------------------------------------------------------------------------

from matplotlib.dates import DateFormatter
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

log_raw_dates = [" ".join(line.split()[1:3]) for line in random_shuf]
log_percents = [float(re.search(r"([0-9.]+)%", line).group(1)) for line in random_shuf]
log_dates = [parse_log_time(log_time) for log_time in log_raw_dates]

plt.plot(log_dates, log_percents)

ax  = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))

ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))

plt.savefig(GRAPH_FILE)
print("Saved as ", GRAPH_FILE)
