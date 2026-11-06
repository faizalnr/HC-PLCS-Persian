# HC-PLCS: Single-text execution and analysis
# No visible ␞N noise marker is used in the final ciphertext.

import hashlib, math, random, time
from collections import Counter

ALPHABET = list("ابتثجچحخدذرزسشصضطظعغفقکگلمنوهی")
DIACRITICS = ["َ","ِ","ُ","ّ"]
PREFIXES = ["","می‌","نمی‌"]
SUFFIXES = ["","ها","تر","ترین"]
SYNONYMS = {"خوب":["عالی","نیکو"],"زیبا":["قشنگ","دلنشین"],
            "سریع":["تند","شتابان"],"کتاب":["دفتر","نوشتار"]}
NOISE_TOKENS = ["و","که","از","در"]
NOISE_THRESHOLD = 0.60

PLAINTEXT = "کتاب خوب است"
KEY = "advanced_key"
NONCE = bytes.fromhex("00112233445566778899aabbccddeeff")

def normalize(s):
    s = str(s).replace("ي","ی").replace("ى","ی").replace("ك","ک").replace("\u0640","")
    return " ".join(s.strip().split())

def seed(key, nonce):
    return int.from_bytes(hashlib.sha256(normalize(key).encode()+b"|"+nonce).digest()[:8],"big")

def lseed(s, label):
    return int.from_bytes(hashlib.sha256(f"{s}|HCPLCS|{label}".encode()).digest()[:8],"big")

def entropy(s):
    if not s: return 0.0
    c=Counter(s); n=len(s)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

def chi2(s):
    if not s: return 0.0
    c=Counter(s); k=len(c)
    if k<=1: return 0.0
    e=len(s)/k
    return sum((v-e)**2/e for v in c.values())

def morph(words, s):
    r=random.Random(s); out=[]; ctl=[]
    for w in words:
        a=r.randrange(len(PREFIXES)); b=r.randrange(len(SUFFIXES))
        out.append(PREFIXES[a]+w+SUFFIXES[b]); ctl.append((a,b))
    return out,ctl

def morph_inv(words, ctl):
    out=[]
    for w,(a,b) in zip(words,ctl):
        if PREFIXES[a]: w=w[len(PREFIXES[a]):]
        if SUFFIXES[b]: w=w[:-len(SUFFIXES[b])]
        out.append(w)
    return out

def synonym(words,s):
    r=random.Random(s); out=[]; ctl=[]
    for w in words:
        opts=SYNONYMS.get(w)
        if opts:
            i=r.randrange(len(opts)); out.append(opts[i]); ctl.append(i)
        else:
            out.append(w); ctl.append(-1)
    return out,ctl

def syntax(words,s):
    r=random.Random(s); order=list(range(len(words))); r.shuffle(order)
    return [words[i] for i in order],order

def syntax_inv(words,order):
    out=[None]*len(words)
    for new,old in enumerate(order): out[old]=words[new]
    return out

def noise(words,s):
    r=random.Random(s); out=[]; mask=[]
    for w in words:
        out.append(w)
        if r.random()>NOISE_THRESHOLD:
            t=r.choice(NOISE_TOKENS); out.append(t); mask.append(t)
        else: mask.append(None)
    return out,mask

def noise_mask(n,s):
    r=random.Random(s); m=[]
    for _ in range(n):
        m.append(r.choice(NOISE_TOKENS) if r.random()>NOISE_THRESHOLD else None)
    return m

def noise_inv(words,mask):
    out=[]; p=0
    for x in mask:
        out.append(words[p]); p+=1
        if x is not None: p+=1
    if p!=len(words): raise ValueError("Noise structure mismatch")
    return out

def diacritics(s,s0):
    r=random.Random(s0); out=[]
    for ch in s:
        out.append(ch)
        if ch in ALPHABET and r.random()>0.50: out.append(r.choice(DIACRITICS))
    return "".join(out)

def strip_diacritics(s):
    return "".join(ch for ch in s if ch not in DIACRITICS)

def graph(s,m):
    return "".join(m.get(ch,ch) for ch in s)

def parameters(key,nonce):
    s=seed(key,nonce)
    r=random.Random(lseed(s,"GRAPHEME"))
    shuffled=ALPHABET[:]; r.shuffle(shuffled)
    return {
        "seed":s,
        "forward":dict(zip(ALPHABET,shuffled)),
        "inverse":dict(zip(shuffled,ALPHABET)),
        "morph":lseed(s,"MORPHOLOGY"),
        "syn":lseed(s,"SYNONYM"),
        "syntax":lseed(s,"SYNTAX"),
        "noise":lseed(s,"NOISE"),
        "dia":lseed(s,"DIACRITIC")
    }

def encrypt(text,key,nonce):
    p=normalize(text); words=p.split(); q=parameters(key,nonce)
    a,mc=morph(words,q["morph"])
    b,sc=synonym(a,q["syn"])
    c,order=syntax(b,q["syntax"])
    d,nm=noise(c,q["noise"])
    noise_stage=" ".join(d)
    dia=diacritics(noise_stage,q["dia"])
    cipher=graph(dia,q["forward"])
    meta={"normalized":p,"morph":a,"morph_ctl":mc,"syn":b,"syn_ctl":sc,
          "syntax":c,"order":order,"noise":d,"noise_mask":nm,
          "diacritics":dia,"word_count":len(words)}
    return cipher,meta

def decrypt(cipher,key,nonce,word_count):
    q=parameters(key,nonce)
    s=graph(cipher,q["inverse"])
    words=strip_diacritics(s).split()
    words=noise_inv(words,noise_mask(word_count,q["noise"]))
    dummy=[None]*word_count
    _,order=syntax(dummy,q["syntax"])
    words=syntax_inv(words,order)
    reverse={}
    for original,opts in SYNONYMS.items():
        for x in opts: reverse[x]=original
    words=[reverse.get(x,x) for x in words]
    _,ctl=morph([""]*word_count,q["morph"])
    return normalize(" ".join(morph_inv(words,ctl)))

if __name__ == "__main__":
    t0=time.perf_counter()
    ciphertext,meta=encrypt(PLAINTEXT,KEY,NONCE)
    enc_ms=(time.perf_counter()-t0)*1000

    t0=time.perf_counter()
    recovered=decrypt(ciphertext,KEY,NONCE,meta["word_count"])
    dec_ms=(time.perf_counter()-t0)*1000

    print("\n"+"="*80)
    print("HC-PLCS SINGLE-TEXT EXECUTION")
    print("="*80)
    print("Plaintext                 :",meta["normalized"])
    print("Key                       :",KEY)
    print("Nonce                     :",NONCE.hex())
    print("\n1. Normalization         :",meta["normalized"])
    print("2. Morphology             :"," ".join(meta["morph"]))
    print("3. Synonym                :"," ".join(meta["syn"]))
    print("4. Syntax permutation    :"," ".join(meta["syntax"]))
    print("   Permutation            :",meta["order"])
    print("5. Noise injection        :"," ".join(meta["noise"]))
    print("   Noise mask             :",meta["noise_mask"])
    print("6. Diacritics             :",meta["diacritics"])
    print("7. Final grapheme         :",ciphertext)
    print("\nDECRYPTION")
    print("Recovered plaintext       :",recovered)
    print("Exact recovery            :",recovered==meta["normalized"])
    pb=len(meta["normalized"].encode()); cb=len(ciphertext.encode())
    print("\nANALYSIS")
    print("Plaintext UTF-8 bytes     :",pb)
    print("Ciphertext UTF-8 bytes    :",cb)
    print(f"EER                       : {cb/max(1,pb):.6f}")
    print(f"Storage overhead          : {(cb-pb)/max(1,pb)*100:.4f}%")
    print(f"Plaintext entropy         : {entropy(meta['normalized']):.6f}")
    print(f"Ciphertext entropy        : {entropy(ciphertext):.6f}")
    print(f"Entropy gain              : {entropy(ciphertext)-entropy(meta['normalized']):+.6f} bits/char")
    print(f"Plaintext chi-square      : {chi2(meta['normalized']):.6f}")
    print(f"Ciphertext chi-square     : {chi2(ciphertext):.6f}")
    print(f"Encryption time            : {enc_ms:.6f} ms")
    print(f"Decryption time            : {dec_ms:.6f} ms")
    print("="*80)
