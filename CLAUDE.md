# ccs1112.github.io

Jekyll blog. Plain kramdown + rouge, MathJax for math, TikZJax for diagrams. Aesthetic: arxiv-style monochrome serif (`Noto Serif`), Computer Modern in figures and math so they share typography.

## Hero diagrams

Every post that explains a non-trivial mechanism gets a hero diagram before the lead paragraph. The diagram must show the whole system in one frame — a reader who lands cold should grasp the shape of the problem from the figure alone.

### Algorithm for making the diagram

1. **Read the entire post.** Every section, every footnote.
2. **Read all linked citations.** Follow every footnote link. Skim the abstract and figures of papers; read the conclusion of issues; read the diff of PRs.
3. **Pay particular attention to the GitHub issue and the GitHub PR, including its code diff.** The issue establishes the failure mode in production terms; the PR's diff shows exactly which lines moved and what the fix was. The diagram must reflect both — what was wrong, and what the fix is — at the granularity of the actual change.
4. **Understand the entire mechanism.** Be able to explain it back without referring to the post. If you can't, you don't yet know enough to draw it.
5. **Draw with as much detail as necessary to completely understand all aspects of the post.** Don't trim load-bearing nodes for visual tidiness; trim only what the prose already covers and the figure doesn't need to repeat.

### Mechanics

Tool: TikZ rendered client-side by TikZJax (loaded in `_layouts/post.html`).

Place the figure as the first block after the frontmatter, before the lead paragraph:

```html
<div class="tikz-figure">
<script type="text/tikz">
\usetikzlibrary{arrows.meta,positioning,fit,calc}
\begin{tikzpicture}[ ... ]
  ...
\end{tikzpicture}
</script>
</div>
```

### Style conventions

- Monochrome only. No fill colors. The blog inverts the SVG in dark mode (`filter: invert(0.92) hue-rotate(180deg)` in `style.css`), so any color would invert unpleasantly.
- Code identifiers in `\texttt{...}`. Prose labels in plain text.
- Edge labels use a `edgelbl` style with `fill=white, inner sep=1.5pt` so labels break the line cleanly.
- Hazard / pathological path: `dashed` boxes and edges.
- Fix path: `line width=0.8pt` (thicker) boxes and edges.
- Container groupings: `fit=` with `dotted` (inner store) and `dashed` (outer context) outlines, labeled at `north west`.
- Math symbols (`$\Rightarrow$`, `$\sim$`) must be in math mode.

### TikZJax library notes

TikZJax preloads most common libraries but is conservative — `\usetikzlibrary{arrows.meta,positioning,fit,calc}` covers what the existing Mooncake diagram needs. Adding `shapes.geometric` works if a cylinder or other geometric shape is needed.

## Other conventions

- Posts use kramdown footnote syntax (`[^name]`) for citations. See existing posts for style.
- Commit messages: Chris Beams why-not-how. Don't restate the diff.
