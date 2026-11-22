import json,sys,time,urllib.request
port=int(sys.argv[1]); out=sys.argv[2]
src=open(sys.argv[3]).read(); P=eval(src[src.index('P={'):].split('\ndef ')[0][2:])
def clocks(): return {"mono":time.clock_gettime_ns(time.CLOCK_MONOTONIC),"boot":time.clock_gettime_ns(time.CLOCK_BOOTTIME),"real":time.time_ns()}
R={}
for k in ("code","math"):
    b={"messages":[{"role":"user","content":P[k]}],"max_tokens":400,"temperature":0,"cache_prompt":False,"repetition_penalty":1.0,"chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(b).encode(),{"Content-Type":"application/json"})
    t0=clocks(); o=json.load(urllib.request.urlopen(r,timeout=1800)); t1=clocks()
    R[k]={"text":o["choices"][0]["message"]["content"],"t0":t0,"t1":t1,"usage":o.get("usage")}
json.dump(R,open(out,"w"))
