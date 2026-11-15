import re,sys
t=open(sys.argv[1]).read()
t=t.split('\\appendix')[0]
t=re.sub(r'(?<!\\)%.*','',t)
t=re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}','',t,flags=re.S)
t=re.sub(r'\\begin\{equation\}.*?\\end\{equation\}','',t,flags=re.S)
heads=[(m.start(),m.group(1) or m.group(2)) for m in re.finditer(r'\\(?:section\*?\{([^}]*)\}|begin\{(abstract)\})',t) if m.group(1) not in('enumerate','itemize','document','tabular','aligned')]
heads=[h for h in heads]
out=[]
for i,(s,n) in enumerate(heads):
    e=heads[i+1][0] if i+1<len(heads) else len(t)
    seg=t[s:e]
    seg=re.sub(r'\\(cite[pt]?|ref|label|includegraphics|bibliography)\{[^}]*\}','',seg)
    seg=re.sub(r'\$[^$]*\$','X',seg)
    seg=re.sub(r'\\[a-zA-Z]+\*?','',seg)
    seg=re.sub(r'[{}\[\]~]',' ',seg)
    w=[x for x in seg.split() if re.search(r'[A-Za-z0-9]',x)]
    out.append((n,len(w)))
for n,c in out: print(f'{c:5d} {n}')
