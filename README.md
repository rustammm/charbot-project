>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "Самая высокая гора"}').json()

{'uid': 1, 'from': 'model', 'ok': True, 'reply': 'а ты не понял '}


>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "Самая высокая гора"}').json()

{'uid': 1, 'from': 'wiki', 'ok': True, 'reply': ' К северу от города находится национальный парк «Денали», в котором расположена самая высокая гора в Северной Америке Денали', 'score': 0.013749625271739802}


>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "Привет"}').json()
{'uid': 1, 'from': 'wiki', 'ok': True, 'reply': '<br>Примите мой привет и благодарность Красной Армии', 'score': 0.003316840651527377}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "Привет"}').json()
{'uid': 1, 'from': 'model', 'ok': True, 'reply': 'я не я '}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "это очень плохой фильм"}').json()
{'uid': 1, 'from': 'model', 'ok': True, 'reply': 'да '}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "это очень плохой фильм ! "}').json()
{'uid': 1, 'from': 'model', 'ok': True, 'reply': 'а ты не такои '}

>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "Почему все так сложно"}').json()
{'uid': 1, 'from': 'wiki', 'ok': True, 'reply': 'Массу звёзд в системе сложно установить, не зная с точностью все элементы орбиты', 'score': 0.004169726192116791}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "Где растут деревья в америке"}').json()
{'uid': 1, 'from': 'wiki', 'ok': True, 'reply': 'Центральная Америка обладает богатейшими лесами, где растут деревья ценных твёрдых пород, таких как красное дерево', 'score': 0.013913974837012986}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "когда в россии выборы"}').json()
{'uid': 1, 'from': 'model', 'ok': True, 'reply': 'это не нельзя '}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "когда в россии выборы"}').json()
{'uid': 1, 'from': 'wiki', 'ok': True, 'reply': ' Если бы мы не провели залоговую приватизацию, то коммунисты выиграли бы выборы в 1996 году, и это были бы последние свободные выборы в России, потому что эти ребята так просто власть не отдают»', 'score': 0.008879210952706874}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "ты не прав"}').json()
{'uid': 1, 'from': 'model', 'ok': True, 'reply': 'ты не по изучил '}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "почему в россии так много нефти"}').json()
{'uid': 1, 'from': 'wiki', 'ok': True, 'reply': 'В России первое письменное упоминание о получении нефти появилось в XVI веке', 'score': 0.006259389191888187}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "Где в спарте можно поесть"}').json()
{'uid': 1, 'from': 'wiki', 'ok': True, 'reply': ' Кроме хмельных напитков пандарены любят поесть и поустраивать поединки (дружеские, естественно), ведь они\xa0— прирождённые мастера боевых искусств, таких как кунг-фу', 'score': 0.0035993654089332156}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "Сколько я тебе должен"}').json()
{'uid': 1, 'from': 'wiki', 'ok': True, 'reply': ' В отместку негодующий Сальвадор Дали послал отцу в конверте свою сперму с гневным письмом: «Это всё, что я тебе должен»', 'score': 0.01386193359181636}

>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "пойдем играть во двор"}').json()
{'uid': 1, 'from': 'wiki', 'ok': True, 'reply': ' После уточнения, будет ли он играть ковбоя, Дилан парировал: «Нет, я буду играть свою мать»', 'score': 0.004930263379235835}
3
>>> r.get('http://127.0.0.1:51000/api/{"uid": 1, "query": "пойдем играть во двор"}').json()
{'uid': 1, 'from': 'model', 'ok': True, 'reply': 'а ты не такои '}

Чтобы запустить серсис нужно скучать https://yadi.sk/d/vIe8vtAXw-eTfQ и положить в директорию service директорию data и вызвать ./start.sh
