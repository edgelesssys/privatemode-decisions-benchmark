"""A labelling page for the flagged banking77 examples, for a person to check.

    python -m bench.label_page results/calibration/part-1/banking77-label-check.json \\
        label-check.html

Part 1 flagged the examples where most other systems disagree with the gold
label, and an LLM judged them. This writes one self-contained HTML page with
the text and both labels, in random order and without the LLM's verdict so
it can't anchor the reader. Answers stay in the browser until "Export"
downloads them in the same format as the input, which
``bench.calibrate_report --label-check`` reads.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>banking77 label check</title>
<style>
:root { --bg: #fff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7; --pick: #0a66c2; --card: #f5f5f7; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161617; --fg: #f5f5f7; --muted: #a1a1a6; --line: #3a3a3c; --pick: #4c9aff; --card: #1f1f21; }
}
body { background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif;
       max-width: 760px; margin: 0 auto; padding: 16px; }
h1 { font-size: 20px; margin: 8px 0; }
p.lead { color: var(--muted); }
.item { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px;
        margin: 12px 0; }
.text { font-size: 16px; margin-bottom: 8px; }
.labels { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
.labels b { color: var(--fg); font-family: ui-monospace, monospace; font-weight: 600; }
button { font: inherit; border: 1px solid var(--line); background: var(--bg); color: var(--fg);
         border-radius: 6px; padding: 6px 10px; margin: 2px 4px 2px 0; cursor: pointer; }
button.on { border-color: var(--pick); color: var(--pick); font-weight: 600; }
.bar { position: sticky; top: 0; background: var(--bg); padding: 8px 0; border-bottom: 1px solid var(--line);
       display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
input { font: inherit; padding: 5px 8px; border: 1px solid var(--line); border-radius: 6px;
        background: var(--bg); color: var(--fg); }
</style></head><body>
<h1>banking77: which label is right?</h1>
<p class="lead">__COUNT__ customer messages where most other systems disagree with the dataset's
label. For each, read the message and pick whether the <b>dataset label</b> is right, the
<b>other label</b> is right, or <b>both</b> are defensible. The two labels are shown in
random order. Your answers stay in this browser until you export them.</p>
<div class="bar"><input id="who" placeholder="Your name" aria-label="Your name">
<span id="done"></span><button id="export">Export answers</button></div>
<div id="items"></div>
<script>
const ROWS = __ROWS__;
const KEY = "banking77-label-check";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
const items = document.getElementById("items");
function render() {
  items.innerHTML = "";
  for (const row of ROWS) {
    const div = document.createElement("div");
    div.className = "item";
    const first = row.swap ? row.consensus : row.gold, second = row.swap ? row.gold : row.consensus;
    div.innerHTML = `<div class="text"></div><div class="labels">A: <b></b> &nbsp; B: <b></b></div>`;
    div.querySelector(".text").textContent = row.text;
    const bs = div.querySelectorAll(".labels b");
    bs[0].textContent = first; bs[1].textContent = second;
    const options = [["A is right", row.swap ? "consensus right" : "gold right"],
                     ["B is right", row.swap ? "gold right" : "consensus right"],
                     ["Both defensible", "both defensible"], ["Neither", "neither"]];
    for (const [label, verdict] of options) {
      const b = document.createElement("button");
      b.textContent = label;
      if (state[row.index] === verdict) b.className = "on";
      b.onclick = () => { state[row.index] = verdict; save(); render(); };
      div.appendChild(b);
    }
    items.appendChild(div);
  }
  const answered = Object.keys(state).filter(k => !k.startsWith("__")).length;
  document.getElementById("done").textContent = `${answered} of ${ROWS.length} done`;
}
document.getElementById("who").value = state.__who || "";
document.getElementById("who").oninput = (e) => { state.__who = e.target.value; save(); };
document.getElementById("export").onclick = () => {
  const out = {checked_by: (state.__who || "a person") + ", reading each text against both labels in random order",
               rule: __RULE__,
               rows: ROWS.filter(r => state[r.index]).map(r => ({index: r.index, text: r.text, gold: r.gold,
                                                                 consensus: r.consensus, verdict: state[r.index]}))};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "banking77-label-check-person.json";
  a.click();
};
render();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("check")
    parser.add_argument("out")
    args = parser.parse_args()
    check = json.loads(Path(args.check).read_text())
    rng = random.Random(0)
    rows = [{"index": r["index"], "text": r["text"], "gold": r["gold"],
             "consensus": r["consensus"], "swap": rng.random() < 0.5} for r in check["rows"]]
    rng.shuffle(rows)
    page = (PAGE.replace("__COUNT__", str(len(rows)))
            .replace("__ROWS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__RULE__", json.dumps(check["rule"])))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page)
    print(f"wrote {args.out} ({len(rows)} examples)")


if __name__ == "__main__":
    main()
