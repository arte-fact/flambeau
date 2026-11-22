# Per-shape H2: pair the i-th Q4_K matmul launch of V2 with the i-th of V3 in the decode window
# (same model, same cycle sequence on `code`), group by V2 rows (from grid) and k cluster.
# usage: d2b.py <diagdir> tagV2 tagV3 [prompt]
import sys,csv,glob,json,re,statistics as st,collections
D,t2,t3=sys.argv[1:4]; P=sys.argv[4] if len(sys.argv)>4 else 'code'
ANSI=re.compile(r'\x1b\[[0-9;]*m')
def q4k(tag):
    f=glob.glob(f'{D}/{tag}/**/*kernel_trace.csv',recursive=True)[0]
    req=json.load(open(f'{D}/{tag}.json'))
    ft=[dict(re.findall(r'(\w+)=([\d.]+)',ANSI.sub('',l))) for l in open(f'{D}/{tag}.log',errors='replace') if 'completion.first_token' in l]
    i=('code','math').index(P)
    out=collections.defaultdict(list)
    for x in csv.DictReader(open(f)):
        n=x['Kernel_Name'].lower()
        if 'q4_k' not in n or not re.search(r'mmvq|mmq',n): continue
        s=int(x['Start_Timestamp'])
        for clk in ('boot','mono'):
            a=req[P]['t0'][clk]+int(float(ft[i]['ttft_ms'])*1e6); b=req[P]['t1'][clk]
            if a<=s<b: break
        else: continue
        gx,wx=int(x['Grid_Size_X']),int(x['Workgroup_Size_X'])
        rows=gx//wx*8 if 'mmvq' in n else gx  # row-tile: 8 rows/block; x8 MMQ: grid x = rows
        out[x['Agent_Id']].append((s,x['Kernel_Name'],rows,(int(x['End_Timestamp'])-s)/1e3))
    return {a:sorted(v) for a,v in out.items()}
A,B=q4k(t2),q4k(t3)
for ag in sorted(A):
    a,b=A[ag],B.get(ag,[])
    print(f'## {P} {ag}: {len(a)} V2 launches, {len(b)} V3 launches')
    if len(a)!=len(b): print('count mismatch, pairing truncated')
    g=collections.defaultdict(list)
    for (s2,n2,r2,d2),(s3,n3,r3,d3) in zip(a,b):
        if r2!=r3: continue
        g[(r2,n2.split('_q8_1')[0][-12:],n3.split('_q8_1')[0][-20:])].append((d2,d3))
    print('| rows | V2 noyau | V3 noyau | lanc. | groupe k | V2 µs | V3 µs | Δ total µs |\n|---|---|---|---|---|---|---|---|')
    for (r,k2,k3),pairs in sorted(g.items()):
        # split by V2 duration into k clusters at gaps > 25 %
        pairs.sort(); cl=[[pairs[0]]]
        for p in pairs[1:]:
            (cl[-1].append(p) if p[0]<=cl[-1][-1][0]*1.25 else cl.append([p]))
        for j,c in enumerate(cl):
            m2=st.median(x for x,_ in c); m3=st.median(y for _,y in c)
            print(f'| {r} | {k2} | {k3} | {len(c)} | {j} | {m2:.1f} | {m3:.1f} | {sum(y-x for x,y in c):+.0f} |')
