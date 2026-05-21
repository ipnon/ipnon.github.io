# ccs1112.github.io

Jekyll blog. Plain kramdown + rouge, MathJax for math, TikZJax for diagrams. Aesthetic: arxiv-style monochrome serif (`Noto Serif` body, Computer Modern in figures and math so they share typography).

This file is loaded automatically by Claude Code whenever a file under `ccs1112.github.io/` is read or written, regardless of the working directory Claude was invoked from. Invoking from `~/Documents/noetics/` and touching a blog file pulls these instructions into context just as invoking from `~/Documents/noetics/ccs1112.github.io/` would. No manual loading needed.

## Post titles

Three directives, all load-bearing:

- **Concise.** Drop articles (`A`, `An`, `The`). Strip every word the title can do without.
- **Precise.** Name the project, name the subsystem if it isn't obvious from the project, and name the bug at its categorical level (the root-cause pathology), not just the surface symptom.
- **Full sentence.** The title must be a grammatically complete sentence with a subject, a verb, and an object — not a noun phrase with a colon-subtitle. The reader should be able to read the title aloud and have it stand on its own.

Canonical shape: `<Root-cause pathology> <action verb> <Project's affected subsystem>`. The pathology is the subject of the sentence; the verb is what it does to the project (`Breaks`, `Crashes`, `Leaks`, `Stalls`, `Corrupts`); the object names the specific subsystem. Use the en-dash (`–`) for compound modifiers (`Allocator–Consumer`, `Forward–Step`), not a hyphen. Title case throughout.

Examples that fit the convention:

- `Drain-Coupling Leaks Mooncake's RDMA QPs Under Peer Failure`
- `Allocator–Consumer Mismatch Crashes FlashInfer's W4A8 Autotune`
- `D-Parameter Shape Mismatch Breaks Mamba2's Forward–Step Parity`

## Hero diagrams

Every post that explains a non-trivial mechanism gets **two diagrams**:

- **Figure 1** at the top, immediately after frontmatter, showing the **broken state** — the system as it was before the fix, with the pathological edge highlighted (dashed) and a hazard sticky listing what goes wrong.
- **Figure 2** at the bottom, after the Conclusion and before the footnotes, showing the **fixed state** — the same layout, the same nodes, with the new edge added (thick) and the hazard sticky removed.

Both figures must use identical node positions so the reader can diff them at a glance. The narrative reads: see broken system → read analysis → see fixed system.

### Algorithm for making the figure

1. **Read the entire post.** Every section, every footnote.
2. **Read all linked citations.** Follow every footnote link. Skim the abstract and figures of papers; read the conclusion of issues; read the diff of PRs.
3. **Pay particular attention to the GitHub issue and the GitHub PR, including its code diff.** The issue establishes the failure mode in production terms; the PR's diff shows exactly which lines moved and what the fix was. The figures must reflect both — what was wrong, and what changed — at the granularity of the actual code.
4. **Understand the entire mechanism.** Be able to explain it back without referring to the post. If you can't, you don't yet know enough to draw it.
5. **Draw with as much detail as necessary to completely understand all aspects of the post.** Don't trim load-bearing nodes for visual tidiness; trim only what the prose already covers and the figure doesn't need to repeat.

### Mechanics

Tool: TikZ rendered client-side by TikZJax (loaded in `_layouts/post.html`).

Place each figure inside:

```html
<div class="tikz-figure">
<script type="text/tikz">
\usetikzlibrary{arrows.meta,positioning,fit,calc}
\begin{tikzpicture}[ ... ]
  ...
\end{tikzpicture}
</script>
<div class="caption"><b>Figure N.</b> One sentence describing what the reader should take away.</div>
</div>
```

### Style conventions

- Monochrome only. No fill colors. The blog inverts the SVG in dark mode (`filter: invert(0.92) hue-rotate(180deg)` in `style.css`), so any color inverts unpleasantly.
- Code identifiers in `\texttt{...}`. Descriptive prose in `\emph{...}`.
- Edge labels use the `edgelbl` style with `fill=white, inner sep=2pt` so labels break the line cleanly.
- Hazard / pathological path: `dashed` boxes and edges.
- Fix path: `line width=0.8pt` (thicker) boxes and edges.
- Container groupings: `fit=` with `dotted` (inner store) and `dashed` (outer context) outlines, labeled at the inside of `north west`.
- Math symbols (`$\Rightarrow$`, `$\sim$`) must be in math mode.
- **No overlapping**: spread nodes generously, prefer `above`/`below`/`left`/`right` positioning over `sloped` where one edge is roughly horizontal or vertical, route long edges with `to[bend ...]` to clear other nodes, and lay out the picture wide so `width: 100%` in CSS expands it to fill the column without losing proportion.

### TikZJax library notes

TikZJax preloads most common libraries but is conservative — `\usetikzlibrary{arrows.meta,positioning,fit,calc}` covers what the current diagrams need. Adding `shapes.geometric` works if a cylinder or other geometric shape is required.

## Other conventions

- Posts use kramdown footnote syntax (`[^name]`) for citations. See existing posts for style.
- Commit messages: Chris Beams why-not-how. Don't restate the diff.
