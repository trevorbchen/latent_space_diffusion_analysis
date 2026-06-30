import json, sys
fp = sys.argv[1]
m = [json.loads(l) for l in open(fp)]
hit = next((x for x in m if x['memorization_fraction'] > 0.01), None)
if hit:
    print('tau_mem step =', hit['step'], 'mem =', round(hit['memorization_fraction'], 4))
else:
    print('tau_mem: never')
print('current step =', m[-1]['step'], 'final mem =', round(m[-1]['memorization_fraction'], 4),
      'max mem =', round(max(x['memorization_fraction'] for x in m), 4))
