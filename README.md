# afterimage

A single dark page. One line, and the light it leaves behind.

## Run it

```bash
pnpm install
pnpm dev
```

Then open http://localhost:3000.

## Layers

`gradient-background.tsx` — a `GrainGradient` shader from
`@paper-design/shaders-react`, drifting orange over black under a 20% black
overlay.

`app/page.tsx` — the line, centred, fading up out of a blur once on load.

Nothing else, and nothing to click.

## Stack

Next.js 14 (App Router), TypeScript, Instrument Serif, Tailwind (preflight only).
