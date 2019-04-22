import sys
import flask
import json
import random
import requests
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from flask import request, jsonify



app = flask.Flask(__name__)
app.config["DEBUG"] = True

with open('config.json') as f:
    CONFIG = json.load(f)

MODEL_ADDR = CONFIG['model']
WIKI_ADDRS = CONFIG['wiki']
SLAVES_ADDRS = [MODEL_ADDR] + list(WIKI_ADDRS.values())

async def fetch(request, port):
    url = "http://127.0.0.1:{}/api/{}".format(port, request)
    print('Starting {}'.format(url))
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            r = await resp.content.read()
            return r


def fetch_async(request):
    futures = [fetch(request, port) for port in SLAVES_ADDRS]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    #asyncio.set_event_loop(loop)
    results = loop.run_until_complete(asyncio.wait(futures))
    return results


@app.route('/api/<string:request>', methods=['GET'])
def home(request):
    req = json.loads(request, encoding='utf8')
    result = fetch_async(request)
    results = []
    best_wiki = {}
    model_answer = {}
    for t in result:
        for x in t:
            try:
                results += [x.result().decode('utf8')]
                x = json.loads(results[-1])
                if x['ok'] == False : continue
                if x['from'] == 'wiki' and (not best_wiki or x['score'] > best_wiki['score']):
                    best_wiki = x
                if x['from'] == 'model':
                    model_answer = x
            except Exception:
                 pass
    print(results)

    if not best_wiki or not model_answer:
        best = best_wiki or model_answer
    else:
        best = best_wiki if random.random() > 0.4 else model_answer

    return json.dumps(best, ensure_ascii=False)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def main():
    app.run(port=CONFIG['main'])

main()
