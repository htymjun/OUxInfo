# Plan: Match GitHub Pages Appearance to Logo

**Date:** 2026-06-03

## Context

The logo (`docs/img/OUxInfo.png`) has a dark warm-charcoal background (~`#1c1a18`) with champagne/cream lettering (~`#d4c5a0`). The current site uses indigo as both primary and accent colors, and defaults to the light scheme. The goal is to make the site's color scheme and branding reflect the logo's dark, elegant aesthetic.

## What Changes

### 1. `mkdocs.yml`

- **Put `slate` (dark) first** so dark mode is the default
- **Change `primary: indigo` → `primary: custom`** on both palette entries so CSS can control the exact color
- **Change `accent: indigo` → `accent: custom`** on both palette entries
- **Add `logo: img/OUxInfo.png`** so the logo appears in the top-left nav bar
- **Add `favicon: img/OUxInfo.png`** for the browser tab icon

```yaml
theme:
  name: material
  logo: img/OUxInfo.png
  favicon: img/OUxInfo.png
  palette:
    - scheme: slate          # dark first → default
      primary: custom
      accent: custom
      toggle:
        icon: material/brightness-7
        name: Switch to light mode
    - scheme: default
      primary: custom
      accent: custom
      toggle:
        icon: material/brightness-4
        name: Switch to dark mode
```

### 2. `docs/css/extra.css`

Add CSS variable overrides to apply the logo palette:

| Variable | Value | Role |
|---|---|---|
| `--md-primary-fg-color` | `#1c1a18` | nav/header background |
| `--md-primary-fg-color--light` | `#2a2826` | hover variant |
| `--md-primary-fg-color--dark` | `#0f0d0b` | pressed variant |
| `--md-primary-bg-color` | `#d4c5a0` | text/icons on nav bar |
| `--md-primary-bg-color--light` | `rgba(212,197,160,0.7)` | subdued nav text |
| `--md-accent-fg-color` | `#d4c5a0` | links, highlights (dark scheme) |
| `--md-accent-fg-color--transparent` | `rgba(212,197,160,0.1)` | hover overlay |

For the **light scheme** (`[data-md-color-scheme="default"]`), use a darker champagne for links so they are readable on a white background:
- `--md-accent-fg-color: #8b7355`
- `--md-typeset-a-color: #8b7355`

## Files Modified

| File | Action |
|---|---|
| `mkdocs.yml` | Swap palette order, set primary/accent to custom, add logo + favicon |
| `docs/css/extra.css` | Add color variable overrides |

## Verification

1. Run `mkdocs serve` locally
2. Confirm: nav bar is dark charcoal, links/highlights are champagne
3. Confirm: dark mode is the default on first load
4. Confirm: logo appears in the top-left nav area
5. Confirm: toggle switches between dark and light modes without broken colors
