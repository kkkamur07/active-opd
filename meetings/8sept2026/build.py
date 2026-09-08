import base64, re
src = open("meetings/next/slides.html").read()
head = src[:src.index('<div class="deck-viewport">')]
tail = src[src.index('  </main>'):]
head = head.replace("<title>APOD Background</title>", "<title>APOD Results 8 Sept</title>")
head = head.replace("</style>", """
.img-fig img { width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 4px; background: #fff; }
.big-q { font-family: var(--font-display); font-size: 64px; line-height: 1.15; color: var(--text); max-width: 1500px; margin: 60px 0 40px; }
.big-q em { color: var(--accent); font-style: italic; }
.bul { font-size: 21px; color: var(--text-dim); line-height: 1.6; max-width: 1500px; }
.bul li { margin-bottom: 12px; padding-left: 6px; }
.bul b { color: var(--text); font-weight: 600; }
.bul .mono { font-family: var(--font-mono); font-size: 18px; color: var(--text); }
.cfg-table { width: 100%; border-collapse: collapse; font-size: 17px; }
.cfg-table th { font-family: var(--font-mono); font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-dim); font-weight: 500; text-align: left; padding: 0 14px 10px 0; border-bottom: 1px solid var(--line-strong); }
.cfg-table td { padding: 12px 14px 12px 0; border-bottom: 1px solid var(--line); vertical-align: top; color: var(--text-dim); line-height: 1.45; }
.cfg-table td b { color: var(--text); font-weight: 600; }
.cfg-table td.run { font-family: var(--font-display); font-size: 26px; color: var(--accent); white-space: nowrap; }
.cfg-table .mono { font-family: var(--font-mono); font-size: 15px; color: var(--text); }
.res-grid { display: grid; grid-template-columns: 1.55fr 0.8fr; gap: 26px; align-items: start; }
/* no-JS fallback: show everything, slides stacked, so the deck is readable in any viewer */
html:not(.js) .reveal { opacity: 1 !important; transform: none !important; }
html:not(.js), html:not(.js) body { overflow: auto !important; height: auto !important; }
html:not(.js) .deck-viewport { position: static !important; overflow: visible !important; }
html:not(.js) .deck-stage { position: static !important; transform: none !important; width: 1920px; height: auto !important; margin: 0 auto; }
@media (max-width: 1850px) { html:not(.js) .deck-stage { zoom: 0.96; } }
@media (max-width: 1750px) { html:not(.js) .deck-stage { zoom: 0.91; } }
@media (max-width: 1650px) { html:not(.js) .deck-stage { zoom: 0.86; } }
@media (max-width: 1550px) { html:not(.js) .deck-stage { zoom: 0.8; } }
@media (max-width: 1450px) { html:not(.js) .deck-stage { zoom: 0.75; } }
@media (max-width: 1350px) { html:not(.js) .deck-stage { zoom: 0.7; } }
@media (max-width: 1250px) { html:not(.js) .deck-stage { zoom: 0.65; } }
@media (max-width: 1150px) { html:not(.js) .deck-stage { zoom: 0.59; } }
@media (max-width: 1050px) { html:not(.js) .deck-stage { zoom: 0.54; } }
@media (max-width: 950px) { html:not(.js) .deck-stage { zoom: 0.49; } }
@media (max-width: 850px) { html:not(.js) .deck-stage { zoom: 0.44; } }
@media (max-width: 750px) { html:not(.js) .deck-stage { zoom: 0.39; } }
@media (max-width: 650px) { html:not(.js) .deck-stage { zoom: 0.33; } }
@media (max-width: 550px) { html:not(.js) .deck-stage { zoom: 0.28; } }
@media (max-width: 450px) { html:not(.js) .deck-stage { zoom: 0.23; } }
html:not(.js) .slide { position: relative !important; visibility: visible !important; opacity: 1 !important; pointer-events: auto !important; margin-bottom: 24px; }
html:not(.js) .deck-controls, html:not(.js) .edit-hotzone, html:not(.js) .edit-toggle { display: none !important; }
</style>""")

def img(name):
    b = base64.b64encode(open(f"meetings/8sept2026/charts/{name}.png","rb").read()).decode()
    return f'<div class="img-fig"><img src="data:image/png;base64,{b}" alt="{name}"></div>'

N = 8
def mast(i, label):
    tail = f' &middot; {label}' if label else ''
    return f'<div class="masthead"><span>Active On-Policy Distillation &middot; Results</span><span class="idx"><b>{i:02d}</b> / {N:02d}{tail}</span></div>'

S = []
# 1 title
S.append(f'''<section class="slide active">{mast(1,"")}
<div class="slide-eyebrow reveal d1">8 September 2026</div>
<div class="big-q reveal d2">Active on-policy distillation</div>
<p class="bul reveal d3">Student Qwen3.5-2B, teacher Qwen3.5-9B. Four experiments, twelve arms, 100 training steps each. MATH-500.</p>
</section>''')
# 2 OPD intro
S.append(f'''<section class="slide">{mast(2,"What is OPD")}
<div class="slide-eyebrow reveal d1">Background</div>
<h2 class="reveal d2" style="margin-bottom:34px;">On-policy distillation: the <em class="t">teacher</em> grades the <em>student's own</em> rollouts</h2>
<div class="two-col even">
<div class="panel reveal d3"><h4>How it works</h4><ul class="bul">
<li>The <b>student</b> (2B) writes a <b>rollout</b> for a question.</li>
<li>The <b>teacher</b> (9B, frozen) scores every token of it.</li>
<li>Loss: per-token <b>reverse KL</b>( &pi;<sub>S</sub> &Vert; &pi;<sub>T</sub> ).</li>
<li>No train/test mismatch: the text is the student's own.</li>
</ul></div>
<div class="panel emph reveal d4"><h4 class="a">Why it is expensive</h4><ul class="bul">
<li>Every rollout costs a student generation plus a teacher pass, up to 8192 tokens.</li>
<li>Every rollout is trained on, whether or not it teaches anything.</li>
<li>MATH-500 avg@4: student <span class="mono">0.27</span>, teacher <span class="mono">0.61</span>.</li>
</ul></div></div>
</section>''')
# 3 our question: which rollouts
S.append(f"""<section class="slide">{mast(3,"Our question")}
<div class="slide-eyebrow reveal d1">Our question</div>
<h2 class="reveal d2" style="margin-bottom:30px;"><em>Which rollouts should we train on?</em></h2>
<div class="two-col even">
<div class="panel reveal d3"><h4 class="t">Token level &middot; prior work</h4>
<div class="fig"><svg viewBox="0 0 680 200" role="img" aria-label="One rollout with a few tokens highlighted">
<text x="40" y="60" font-size="15" fill="var(--text-faint)">one rollout, tokens left to right</text>
<rect x="40" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/><rect x="80" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/><rect x="120" y="120" width="34" height="26" rx="4" fill="var(--teacher)" opacity="0.9"/><rect x="160" y="120" width="34" height="26" rx="4" fill="var(--teacher)" opacity="0.9"/><rect x="200" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/><rect x="240" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/><rect x="280" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/><rect x="320" y="120" width="34" height="26" rx="4" fill="var(--teacher)" opacity="0.9"/><rect x="360" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/><rect x="400" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/><rect x="440" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/><rect x="480" y="120" width="34" height="26" rx="4" fill="var(--teacher)" opacity="0.9"/><rect x="520" y="120" width="34" height="26" rx="4" fill="var(--teacher)" opacity="0.9"/><rect x="560" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/><rect x="600" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/><rect x="640" y="120" width="34" height="26" rx="4" fill="none" stroke="var(--line-strong)" stroke-width="1.5"/>
<text x="40" y="185" class="body" font-size="18" fill="var(--text-dim)">keep or reweight individual tokens inside a rollout</text>
</svg></div>
<p class="caveat" style="margin-top:14px;">A lot of literature focuses on this.</p></div>
<div class="panel emph reveal d4"><h4 class="a">Trajectory level &middot; this work</h4>
<div class="fig"><svg viewBox="0 0 680 340" role="img" aria-label="Several rollouts of one question, some kept for training">
<text x="40" y="50" font-size="15" fill="var(--text-faint)">rollouts of one question</text>
<rect x="40" y="70" width="520" height="30" rx="5" fill="none" opacity="1" stroke="var(--line-strong)" stroke-width="1.5"/><text x="580" y="91" font-size="15" fill="var(--text-faint)">skip</text><rect x="40" y="114" width="520" height="30" rx="5" fill="var(--accent)" opacity="0.9" stroke="var(--accent)" stroke-width="1.5"/><text x="580" y="135" font-size="15" fill="var(--accent)">train</text><rect x="40" y="158" width="520" height="30" rx="5" fill="none" opacity="1" stroke="var(--line-strong)" stroke-width="1.5"/><text x="580" y="179" font-size="15" fill="var(--text-faint)">skip</text><rect x="40" y="202" width="520" height="30" rx="5" fill="var(--accent)" opacity="0.9" stroke="var(--accent)" stroke-width="1.5"/><text x="580" y="223" font-size="15" fill="var(--accent)">train</text><rect x="40" y="246" width="520" height="30" rx="5" fill="none" opacity="1" stroke="var(--line-strong)" stroke-width="1.5"/><text x="580" y="267" font-size="15" fill="var(--text-faint)">skip</text><rect x="40" y="290" width="520" height="30" rx="5" fill="none" opacity="1" stroke="var(--line-strong)" stroke-width="1.5"/><text x="580" y="311" font-size="15" fill="var(--text-faint)">skip</text>
</svg></div>
<p class="caveat" style="margin-top:14px;">Pick whole rollouts, and pick which questions to roll out.</p></div>
</div>
</section>""")

# 3 experiments + goals
S.append(f"""<section class="slide">{mast(4,"Experiments")}
<div class="slide-eyebrow reveal d1">The experiments</div>
<h2 class="reveal d2" style="margin-bottom:40px;">Can <em>selection</em> beat random?</h2>
<div class="panel reveal d3"><ul class="bul" style="font-size:22px;">
<li><b>Rollout selection</b> (kl50): 12 rollouts per question, keep 4 by reverse KL, entropy, or random.</li>
<li><b>Correctness</b> (r1): questions where teacher / student are right or wrong.</li>
<li><b>Uncertainty</b> (r3, r4): most uncertain, most certain, or random questions.</li>
</ul></div>
</section>""")
# 4 configs
S.append(f"""<section class="slide">{mast(5,"Configs")}
<div class="slide-eyebrow reveal d1">Setup</div>
<h2 class="reveal d2" style="margin-bottom:10px;">Setup</h2>
<p class="reveal d2" style="font-size:22px;color:var(--text-dim);margin-bottom:22px;"><b style="color:var(--teacher);">Teacher ceiling: avg@4 0.61</b> &nbsp;&middot;&nbsp; <b>max 8192 tokens per rollout</b> &nbsp;&middot;&nbsp; base student avg@4 0.27, pass@4 0.39</p>
<div class="panel reveal d3"><table class="cfg-table">
<thead><tr><th>Run</th><th>Arms</th><th>Question source</th><th>Rollouts per question</th><th>Trained per arm</th></tr></thead>
<tbody>
<tr><td class="run">kl50</td><td><span class="mono">kl_mid &middot; kl_high &middot; kl_low &middot; random &middot; entropy</span></td><td>136 questions per selection round</td><td><b>12</b>, keep 4 by rule</td><td>50 training steps</td></tr>
<tr><td class="run">r1</td><td><span class="mono">teacher_right_student_wrong &middot; both_right &middot; both_wrong &middot; mixed</span></td><td>800 questions per bucket from a 14,500-question bank</td><td><b>4</b>, all trained</td><td>3,200 rollouts, 100 training steps</td></tr>
<tr><td class="run">r3 / r4</td><td><span class="mono">uncertain_questions &middot; random_questions &middot; certain_questions</span></td><td>top-800 / random 800 / bottom-800 by student H(q)</td><td><b>4</b>, all trained</td><td>3,200 rollouts, 100 training steps</td></tr>
</tbody></table></div>
<div class="common-strip">
<div class="panel reveal d4"><h4>Training</h4><p>Qwen3.5-2B student, Qwen3.5-9B teacher. <b>max 8192 tokens</b> per rollout, effective batch <b>32</b>, peak LR <b>3.16e-6</b>, warmup then cosine to step 100, one schedule per run.</p></div>
<div class="panel reveal d5"><h4>Eval every 10 training steps</h4><p><b>MATH-500</b> avg@4 + pass@4, all 500 questions. A response without a \\boxed{{}} answer counts as wrong.</p></div>
<div class="panel reveal d6"><h4 class="t">Teacher ceiling</h4><p>Qwen3.5-9B, same eval: <b>avg@4 0.61</b>. Base student: <b>avg@4 0.27, pass@4 0.39</b>.</p></div>
</div>
</section>""")
# 5 rollout selection: kl50 only
S.append(f"""<section class="slide">{mast(6,"Rollout selection")}
<div class="slide-eyebrow reveal d1">Result &middot; rollout selection, 12 keep 4</div>
<h2 class="reveal d2" style="margin-bottom:22px;">Rollout selection by <em>distribution overlap</em></h2>
<div class="res-grid">
<div class="panel emph reveal d3"><h4 class="a">kl50 &middot; 50 training steps</h4>{img("chart_kl50")}</div>
<div class="panel reveal d4"><h4>Best avg@4 / best pass@4 (step in brackets)</h4><table class="gap-table" style="font-size:17px;"><thead><tr><th>arm</th><th>avg@4</th><th>pass@4</th></tr></thead><tbody>
<tr><td>kl_mid</td><td>0.475 (34)</td><td>0.612 (34)</td></tr><tr><td>kl_high</td><td>0.432 (34)</td><td>0.562 (34)</td></tr>
<tr><td>random</td><td>0.430 (17)</td><td>0.570 (17)</td></tr><tr><td>entropy</td><td>0.428 (12)</td><td>0.560 (12)</td></tr><tr><td>kl_low</td><td>0.384 (34)</td><td>0.516 (34)</td></tr></tbody></table></div></div>
</section>""")
# 6 r1 correctness
S.append(f"""<section class="slide">{mast(7,"Correctness")}
<div class="slide-eyebrow reveal d1">Result &middot; question selection by correctness</div>
<h2 class="reveal d2" style="margin-bottom:22px;">Question selection by <em>correctness</em></h2>
<div class="res-grid"><div class="panel reveal d3">{img("chart_r1")}</div>
<div class="panel reveal d4"><h4>Best avg@4 / best pass@4 (step in brackets)</h4><table class="gap-table" style="font-size:17px;">
<thead><tr><th>arm</th><th>avg@4</th><th>pass@4</th></tr></thead><tbody><tr><td>teacher right / student wrong</td><td>0.478 (10)</td><td>0.596 (30)</td></tr><tr><td>both right</td><td>0.463 (30)</td><td>0.588 (30)</td></tr><tr><td>mixed</td><td>0.445 (10)</td><td>0.578 (70)</td></tr><tr><td>both wrong</td><td>0.430 (10)</td><td>0.548 (10)</td></tr><tr><td>random 800 (r3)</td><td>0.447 (10)</td><td>0.574 (100)</td></tr></tbody></table></div></div>
</section>""")
# 7 r3/r4 entropy
S.append(f"""<section class="slide">{mast(8,"Uncertainty")}
<div class="slide-eyebrow reveal d1">Result &middot; question selection by student uncertainty</div>
<h2 class="reveal d2" style="margin-bottom:22px;">Question selection by <em>uncertainty</em></h2>
<div class="res-grid"><div class="panel reveal d3">{img("chart_r3r4")}</div>
<div class="panel reveal d4"><h4>Best avg@4 / best pass@4 (step in brackets)</h4><table class="gap-table" style="font-size:17px;">
<thead><tr><th>arm</th><th>avg@4</th><th>pass@4</th></tr></thead><tbody><tr><td>random 800</td><td>0.447 (10)</td><td>0.574 (100)</td></tr><tr><td>uncertain (top-800 H(q))</td><td>0.430 (90)</td><td>0.552 (70)</td></tr><tr><td>certain (bottom-800 H(q))</td><td>0.461 (10)</td><td>0.580 (10)</td></tr></tbody></table></div></div>
</section>""")

html = head + '<div class="deck-viewport">\n  <main class="deck-stage" id="deckStage">\n' + "\n".join(S) + "\n" + tail
html = html.replace("<script>\n/* ====", "<script>\ndocument.documentElement.classList.add('js');\n/* ====", 1)
html = html.replace("this.loadEdits();", "/* edit restore disabled: always show the built content */")
NAV_JS = """
const deck = new SlidePresentation();
/* focus: the page may sit in an iframe that does not get keyboard focus by default */
document.body.tabIndex = -1;
const grab = () => { try { window.focus(); document.body.focus({ preventScroll: true }); } catch (e) {} };
grab(); window.addEventListener('load', grab); document.addEventListener('pointerdown', grab);
/* click on the stage: left fifth goes back, anywhere else goes forward */
document.querySelector('.deck-viewport').addEventListener('click', (e) => {
  if (e.target.closest('a, button, [contenteditable="true"]')) return;
  if (e.clientX < window.innerWidth * 0.2) deck.showSlide(deck.currentSlide - 1); else deck.showSlide(deck.currentSlide + 1);
});
/* on-screen arrows */
const mk = (txt, dir, side) => { const b = document.createElement('button'); b.textContent = txt; b.setAttribute('aria-label', dir < 0 ? 'previous' : 'next');
  b.style.cssText = 'position:fixed;top:50%;' + side + ':14px;transform:translateY(-50%);z-index:9999;width:44px;height:44px;border-radius:50%;border:1px solid rgba(32,29,23,0.25);background:rgba(255,255,255,0.85);color:#201d17;font-size:22px;cursor:pointer;opacity:0.55;';
  b.onmouseenter = () => b.style.opacity = '1'; b.onmouseleave = () => b.style.opacity = '0.55';
  b.onclick = (e) => { e.stopPropagation(); deck.showSlide(deck.currentSlide + dir); grab(); }; document.body.appendChild(b); };
mk('\u2039', -1, 'left'); mk('\u203a', 1, 'right');
"""
html = html.replace("new SlidePresentation();", NAV_JS, 1)
open("meetings/8sept2026/slides.html", "w").write(html)
print(len(html))
