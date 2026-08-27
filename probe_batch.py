import json, urllib.request, time, threading
BASE='http://127.0.0.1:18020'
def metrics():
    m={}
    for line in urllib.request.urlopen(BASE+'/metrics').read().decode().splitlines():
        if line.startswith('vllm:spec_decode_num_') and '_created' not in line:
            k,v=line.rsplit(' ',1); m[k]=float(v)
    return m
def gen(i, out):
    body=json.dumps({'model':'qwen3.8-27b','prompt':'Explain in detail how a GPU renders a 3D frame, part %d. '%i,'max_tokens':160,'temperature':0,'ignore_eos':True}).encode()
    t0=time.time()
    d=json.loads(urllib.request.urlopen(urllib.request.Request(BASE+'/v1/completions',data=body,headers={'Content-Type':'application/json'})).read())
    out[i]=(d['usage']['completion_tokens'], time.time()-t0, len(d['choices'][0]['text']))
m0=metrics(); t0=time.time(); out={}
ths=[threading.Thread(target=gen,args=(i,out)) for i in range(4)]
[t.start() for t in ths]; [t.join() for t in ths]
wall=time.time()-t0; m1=metrics()
tot=sum(v[0] for v in out.values())
draft=m1['vllm:spec_decode_num_draft_tokens_total']-m0['vllm:spec_decode_num_draft_tokens_total']
acc=m1['vllm:spec_decode_num_accepted_tokens_total']-m0['vllm:spec_decode_num_accepted_tokens_total']
drafts=m1['vllm:spec_decode_num_drafts_total']-m0['vllm:spec_decode_num_drafts_total']
for i in sorted(out): print('req',i,'tokens',out[i][0],'%.1fs'%out[i][1],'len(text)=%d'%out[i][2])
print('--- batched(4): tokens=%d wall=%.1fs throughput=%.1f tok/s'%(tot,wall,tot/wall))
print('drafts=%d draft_tokens=%d accepted=%d acceptance=%.1f%% accept_len=%.2f'%(drafts,draft,acc,100*acc/draft if draft else 0,acc/drafts if drafts else 0))
