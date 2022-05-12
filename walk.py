import json
import matplotlib.pyplot as plt

with open('out.json') as datafile:
    results = json.load(datafile)

datas = results['data']
corners = results['corners']
strengths = {}
ts = []
for sample in datas:
    for signal in sample['signals']:
        if signal['mac'] == mac:
            strengths.append(signal['strength'])
            ts.append(sample['timestamp'])
            break

plt.plot(ts, strengths)
plt.scatter(ts, strengths)
for corner in corners:
    plt.plot((corner, corner), (min(strengths), max(strengths)), color='red')
plt.show()
