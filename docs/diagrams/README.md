# Diagrams

PlantUML sources for the architecture, rendered to SVG (GitHub renders the SVG
in the README). The `.puml` files are the canonical definitions; the SVGs are
generated output.

| file | shows |
| --- | --- |
| `request-path.svg` | the eight-step request chain, trust boundary, both gate layers |
| `gate-two-layer.svg` | layer 1 (deterministic) + layer 2 (semantic), floor merge, blocked stream, artifact |
| `graduation-lifecycle.svg` | bootstrap → export → train → evaluate → package, both graduation tracks |
| `shadow-mode.svg` | `ShadowBackend`: inline primary, background teacher, disagreement accounting |
| `cluster-deployment.svg` | DGX Spark deployment, the boundary between local and egress |

Re-render (any machine with Java 11+ and PlantUML):

```bash
# one-off with a jar:
java -jar plantuml.jar -tsvg docs/diagrams/*.puml

# or via a package manager:
brew install plantuml && plantuml -tsvg docs/diagrams/*.puml
```

Colour convention (matches `docs/architecture.html`): teal = stays local,
oxide red = crosses the boundary, paper background. Keep that when editing —
the colour *is* the security claim.
