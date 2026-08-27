import json, urllib.request, time, sys, threading
BASE='http://127.0.0.1:18020'
mode=sys.argv[1] if len(sys.argv)>1 else 'seq'
def gen(i,out):
    body=json.dumps({'model':'qwen3.8-27b','prompt':'Write a detailed essay about the history of computing, part %d. '%i,'max_tokens':160,'temperature':0,'ignore_eos':True}).encode()
    t0=time.time()
    d=json.loads(urllib.request.urlopen(urllib.request.Request(BASE+'/v1/completions',data=body,headers={'Content-Type':'application/json'})).read())
    out[i]=(d['usage']['completion_tokens'],time.time()-t0,len(d['choices'][0]['text']))
out={}
if mode=='seq':
    for i in range(4): gen(i,out)
else:
    ths=[threading.Thread(target=gen,args=(i,out)) for i in range(4)]
    [t.start() for t in ths]; [t.join() for t in ths]
for i in sorted(out): print('req',i,'tok',out[i][0],'%.2fs'%out[i][1],'tlen',out[i][2])
