import json, urllib.request, time, sys
BASE='http://127.0.0.1:18020'
def metrics():
    m={}
    for line in urllib.request.urlopen(BASE+'/metrics').read().decode().splitlines():
        if line.startswith('vllm:spec_decode_num_') and not '_created' in line:
            k,v=line.rsplit(' ',1); m[k]=float(v)
    return m
def gen(prompt, mt, n):
    body=json.dumps({'model':'qwen3.8-27b','prompt':prompt,'max_tokens':mt,'temperature':0,'ignore_eos':True}).encode()
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request(BASE+'/v1/completions',data=body,headers={'Content-Type':'application/json'}))
    d=json.loads(r.read()); dt=time.time()-t0
    return d['usage']['completion_tokens'], dt
m0=metrics(); t0=time.time(); tot=0
for i in range(4):
    c,dt=gen('Write a detailed essay about the history of computing, part %d. '%i, 160, 0)
    tot+=c; print('req',i,'tokens',c,'%.1fs'%dt)
wall=time.time()-t0; m1=metrics()
draft=m1['vllm:spec_decode_num_draft_tokens_total']-m0['vllm:spec_decode_num_draft_tokens_total']
acc=m1['vllm:spec_decode_num_accepted_tokens_total']-m0['vllm:spec_decode_num_accepted_tokens_total']
drafts=m1['vllm:spec_decode_num_drafts_total']-m0['vllm:spec_decode_num_drafts_total']
print('--- single-stream: tokens=%d wall=%.1fs throughput=%.1f tok/s'%(tot,wall,tot/wall))
print('drafts=%d draft_tokens=%d accepted=%d acceptance=%.1f%% accept_len=%.2f'%(drafts,draft,acc,100*acc/draft if draft else 0,acc/drafts if drafts else 0))
