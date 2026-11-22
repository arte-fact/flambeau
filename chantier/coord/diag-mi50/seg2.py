import sys,csv,re,json,statistics
sys.setrecursionlimit(20000)
kind,trace,log,out=sys.argv[1:5]; G=float(sys.argv[5]) if len(sys.argv)>5 else 3000
ev=[]
if kind=='nsys':
    for x in csv.DictReader(open(trace)):
        if x['GrdX']: s=int(x['Start (ns)']); ev.append((s,s+int(x['Duration (ns)']),x['Name']))
else:
    for x in csv.DictReader(open(trace)):
        if x['Kernel_Name'].startswith('__amd'): continue
        ev.append((int(x['Start_Timestamp']),int(x['End_Timestamp']),x['Kernel_Name']))
ev.sort()
names=[n for *_,n in ev]
gap=[ev[i+1][0]-ev[i][1] for i in range(len(ev)-1)]+[10**12]
cells=[]
for l in open(log):
    m=re.match(r'\s+(C\d)\s+(\S+)\s+rows=(\d+)\s+k=(\d+)\s+m=(\d+)\s+act=',l)
    if m: cells.append(dict(c=m[1],id=m[2],rows=int(m[3]),k=int(m[4]),m=int(m[5])))
N=203
def ok(p,j):
    if p+N*j>len(names): return False
    pat=names[p:p+j]
    if not all(names[p+i]==pat[i%j] for i in range(N*j)): return False
    return gap[p+3*j-1]>=G and gap[p+N*j-1]>=G
memo={}
def solve(ci,p):
    if ci==len(cells): return [] if p==len(names) else None
    if (ci,p) in memo: return memo[(ci,p)]
    r=None
    for j in range(1,6):
        if ok(p,j):
            s=solve(ci+1,p+N*j)
            if s is not None: r=[j]+s; break
    memo[(ci,p)]=r; return r
js=solve(0,0); assert js is not None,'no partition'
p=0
for c,j in zip(cells,js):
    calls=[sum(ev[p+i*j+q][1]-ev[p+i*j+q][0] for q in range(j)) for i in range(3,N)]
    c['kernels']=names[p:p+j]; c['us']=statistics.median(calls)/1000; p+=N*j
json.dump(cells,open(out,'w')); print(len(cells),'ok', sum(1 for j in js if j>1),'multi-kernel cells')
