import sys
import flask
import json
import requests
import asyncio
import aiohttp
import json
from concurrent.futures import ThreadPoolExecutor
from flask import request, jsonify


service = sys.argv[1]

app = flask.Flask(__name__)
app.config["DEBUG"] = True

with open('config.json') as f:
    CONFIG = json.load(f)

MODEL_ADDR = CONFIG['model']
WIKI_ADDRS = CONFIG['wiki']
SLAVES_ADDRS = [MODEL_ADDR] + list(WIKI_ADDRS.values())

CHUNK = CONFIG['wiki_chunk'][service]
print(CHUNK)
import wiki
wiki.prepare(CHUNK)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


@app.route('/api/<string:request>', methods=['GET'])
def home(request):
    req = json.loads(request)
    query = req['query']
    uid = req['uid']

    res = wiki.get_best_doc(query)
    if res:
        doc, is_ok, score = wiki.get_best_doc(query)
    else:
        is_ok = False

    if is_ok:
        return json.dumps({'uid': uid, 'from': 'wiki', 'ok': True, 'reply': doc, 'score': score}, ensure_ascii=False)
    else:
        return json.dumps({'uid': uid, 'from': 'wiki', 'ok': False}, ensure_ascii=False)


def main():
    app.run(port=WIKI_ADDRS[service])

main()
