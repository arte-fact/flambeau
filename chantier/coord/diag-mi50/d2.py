# D2: per-cycle kernel attribution, V2 vs V3, from rocprofv3 kernel traces.
# usage: d2.py <diagdir> tagV2a tagV3a [tagV2b tagV3b ...]   (tags alternate V2,V3)
import sys,csv,glob,json,re,collections,statistics as st
D=sys.argv[1]; tags=sys.argv[2:]
ANSI=re.compile(r'\x1b\[[0-9;]*m')
def fam(n):
    l=n.lower()
    if 'quantize' in l:
        return 'quant_ds4' if re.search(r'mmq|ds4|few',l) else 'quant_q8_1'
    dt=next((d for d in ['iq4_xs','q4_k','q5_k','q6_k','q8_0','q4_0','f16'] if d in l),None)
    if dt and re.search(r'mmvq|mmq|qmatmul|gemv|gemm',l):
        kind='rowtile' if 'row_tile' in l else ('mmq' if re.search(r'mmq|lds_tiled',l) else 'mmvq')
        return f'{dt}_{kind}'
    if re.search(r'attn|attention|flash',l): return 'attention'
    if re.search(r'delta|gdn|ssm|conv1d',l): return 'gdn'
    if re.search(r'rcclkernel|nccl|allreduce|all_reduce',l): return 'comm'
    return 'other'
def load(tag):
    f=glob.glob(f'{D}/{tag}/**/*kernel_trace.csv',recursive=True)
    assert f, f'no trace for {tag}'
    ev=[]
    for x in csv.DictReader(open(f[0])):
        n=x['Kernel_Name']
        if n.startswith('__amd'): continue
        g=(x.get('Grid_Size_X') or x.get('Grid_Size'), x.get('Grid_Size_Y',''), x.get('Workgroup_Size_X') or x.get('Workgroup_Size',''))
        ev.append((int(x['Start_Timestamp']),int(x['End_Timestamp']),x['Agent_Id'],n,g))
    ev.sort()
    req=json.load(open(f'{D}/{tag}.json'))
    log=[ANSI.sub('',l) for l in open(f'{D}/{tag}.log',errors='replace')]
    fin=[dict(re.findall(r'(\w+)=([\d.]+)',l)) for l in log if 'completion.finish' in l]
    ft=[dict(re.findall(r'(\w+)=([\d.]+)',l)) for l in log if 'completion.first_token' in l]
    return ev,req,fin,ft
def window(ev,req,ft,k,i):
    # pick the clock that contains the kernels
    for clk in ('boot','mono'):
        a,b=req[k]['t0'][clk],req[k]['t1'][clk]
        if any(a<=s<=b for s,*_ in ev[:: max(1,len(ev)//2000)]): break
    return a+int(float(ft[i]['ttft_ms'])*1e6), b, clk
rows={}
for tag in tags:
    ev,req,fin,ft=load(tag)
    for i,k in enumerate(('code','math')):
        a,b,clk=window(ev,req,ft,k,i)
        cyc=int(fin[i]['mtp_cycles'])
        sel=[e for e in ev if a<=e[0]<b]
        for agent in sorted({e[2] for e in sel}):
            es=[e for e in sel if e[2]==agent]
            busy=0; end=0
            for s,e,*_ in es:
                if e>end: busy+=e-max(s,end); end=e
            fsum=collections.defaultdict(float); fcnt=collections.Counter()
            grid=collections.defaultdict(list)
            for s,e,_,n,g in es:
                f=fam(n); fsum[f]+=(e-s)/1e3; fcnt[f]+=1
                if f.startswith('q4_k') or f.startswith('quant'): grid[(f,n,g)].append((e-s)/1e3)
            rows[(tag,k,agent)]=dict(cyc=cyc,wall=(b-a)/1e3/cyc,busy=busy/1e3/cyc,launch=len(es)/cyc,
                fsum={f:v/cyc for f,v in fsum.items()},fcnt={f:v/cyc for f,v in fcnt.items()},
                grid={kk:(len(v)/cyc,st.median(v)) for kk,v in grid.items()},clk=clk)
json.dump({'|'.join(k):{**v,'grid':{str(g):x for g,x in v['grid'].items()}} for k,v in rows.items()},open(f'{D}/d2_rows.json','w'))
V2=tags[0::2]; V3=tags[1::2]
def avg(ts,k,ag,key,sub=None):
    vals=[(rows[(t,k,ag)][key] if sub is None else rows[(t,k,ag)][key].get(sub,0)) for t in ts if (t,k,ag) in rows]
    return sum(vals)/len(vals) if vals else float('nan')
agents=sorted({ag for (_,_,ag) in rows})
for k in ('code','math'):
    for ag in agents:
        print(f'\n## {k}, agent {ag} (µs per cycle; V2 {V2} vs V3 {V3})')
        print('| quantité | V2 | V3 | Δ |\n|---|---|---|---|')
        for key in ('wall','busy','launch'):
            x,y=avg(V2,k,ag,key),avg(V3,k,ag,key); print(f'| {key} | {x:.1f} | {y:.1f} | {y-x:+.1f} |')
        print(f'| idle (wall−busy) | {avg(V2,k,ag,"wall")-avg(V2,k,ag,"busy"):.1f} | {avg(V3,k,ag,"wall")-avg(V3,k,ag,"busy"):.1f} | |')
        fams=sorted({f for t in tags for f in rows.get((t,k,ag),{}).get('fsum',{})})
        for f in fams:
            x,y=avg(V2,k,ag,'fsum',f),avg(V3,k,ag,'fsum',f); cx,cy=avg(V2,k,ag,'fcnt',f),avg(V3,k,ag,'fcnt',f)
            if max(x,y)>1: print(f'| {f} (lanc. {cx:.1f}→{cy:.1f}) | {x:.1f} | {y:.1f} | {y-x:+.1f} |')
        print('\n### Q4_K et quantification par noyau/grille (lancements/cycle, médiane µs)')
        for t in (V2[0],V3[0]):
            r=rows.get((t,k,ag));
            if not r: continue
            for (f,n,g),(c,m) in sorted(r['grid'].items(),key=lambda kv:-kv[1][0]*kv[1][1]):
                print(f'- {t} {f} {n[:70]} grid={g} : {c:.2f}/cyc × {m:.1f} µs')
