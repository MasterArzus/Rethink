import textwrap


def render_token_card(token, index, prob, is_critical, reason, is_selected, is_new=False):
    """Deprecated: Use render_token_stream_inline instead."""
    return render_token_stream_inline(
        [{"index": index, "token": token, "prob": prob, "is_critical": is_critical, "reason": reason, "is_new": is_new}],
        selected_idx=index
    )


def render_token_stream_inline(token_data, selected_idx):
    """
    Renders tokens as inline flowing text with color-coded SOS highlighting.
    - Each token is a clickable inline span
    - Background color indicates SOS score (red=high SOS, green=low)
    - Hover shows tooltip with token details
    - Click selects the token
    """
    css = textwrap.dedent("""
    <style>
        .token-stream-container {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            line-height: 2.2;
            padding: 16px;
            background-color: #fafafa;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
        }
        .token-stream-text {
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .token-span {
            display: inline;
            padding: 2px 1px;
            margin: 0 1px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
            position: relative;
        }
        /* Prob-based coloring: white (high prob) → green → yellow → red (low prob) */
        .token-span.prob-white { background-color: rgba(255, 255, 255, 0.6); }
        .token-span.prob-green { background-color: rgba(180, 230, 200, 0.45); }
        .token-span.prob-yellow { background-color: rgba(255, 255, 160, 0.45); }
        .token-span.prob-red { background-color: rgba(255, 180, 180, 0.5); }

        /* Critical flag override */
        .token-span.critical { border-bottom: 2px solid #e91e63; }

        /* Selected state */
        .token-span.selected {
            background-color: #2196f3 !important;
            color: white;
            box-shadow: 0 2px 8px rgba(33, 150, 243, 0.4);
        }

        /* New tokens (after rewind) */
        .token-span.new-token {
            border-bottom: 2px dashed #2196f3;
        }

        /* Hover tooltip */
        .token-span:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            white-space: nowrap;
            z-index: 1000;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            font-family: 'Segoe UI', monospace;
        }
        .token-span:hover::before {
            content: '';
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 4px solid transparent;
            border-bottom-color: #333;
            z-index: 1000;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }
        .token-span:hover::after,
        .token-span:hover::before {
            opacity: 1;
        }

        /* Selected token tooltip override */
        .token-span.selected:hover::after {
            background-color: #1565c0;
        }

        /* Anchor (a) tag resets for inline tokens */
        .token-span {
            text-decoration: none !important;
            color: inherit !important;
        }

        /* SOS Legend */
        .sos-legend {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            font-size: 11px;
            color: #666;
        }
        .sos-legend-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .sos-color-box {
            width: 14px;
            height: 14px;
            border-radius: 3px;
        }
    </style>
    """)

    # Build SOS legend
    legend = textwrap.dedent("""
    <div class="sos-legend">
        <span>SOS Level:</span>
        <div class="sos-legend-item"><div class="sos-color-box" style="background-color: rgba(255, 255, 255, 0.6); border: 1px solid #ddd;"></div> Normal</div>
        <div class="sos-legend-item"><div class="sos-color-box" style="background-color: rgba(180, 230, 200, 0.45);"></div> Low</div>
        <div class="sos-legend-item"><div class="sos-color-box" style="background-color: rgba(255, 255, 160, 0.45);"></div> Medium</div>
        <div class="sos-legend-item"><div class="sos-color-box" style="background-color: rgba(255, 180, 180, 0.5);"></div> High</div>
        <span style="margin-left: 12px;">|</span>
        <div class="sos-legend-item"><div style="width:14px; height:14px; border-bottom: 2px solid #e91e63; border-radius: 3px;"></div> Critical</div>
        <div class="sos-legend-item"><div style="width:14px; height:14px; border-bottom: 2px dashed #2196f3; border-radius: 3px;"></div> New</div>
    </div>
    """)

    # Build token spans
    token_spans = []
    for item in token_data:
        idx = item['index']
        token_text = item['token']
        prob = item['prob']
        is_critical = item['is_critical']
        reason = item['reason']
        is_new = item.get('is_new', False)
        is_selected = (idx == selected_idx)

        # Map prob to color class (high prob = white, low prob = red)
        if prob >= 0.75:
            prob_class = "prob-white"
        elif prob >= 0.5:
            prob_class = "prob-green"
        elif prob >= 0.25:
            prob_class = "prob-yellow"
        else:
            prob_class = "prob-red"

        # Escape HTML
        safe_token = token_text.replace("<", "&lt;").replace(">", "&gt;")
        if safe_token.strip() == "":
            # Preserve whitespace
            token_spans.append(safe_token)
            continue

        # Build classes
        classes = [prob_class]
        if is_critical:
            classes.append("critical")
        if is_selected:
            classes.append("selected")
        if is_new:
            classes.append("new-token")

        # Build tooltip
        tooltip = f"Token: '{safe_token}' | Prob: {prob:.4f}"
        if is_critical:
            tooltip += f" | Critical: {reason}"
        tooltip = tooltip.replace('"', '&quot;')

        # Use <a> tag for clickable token (st_click_detector requires clickable elements with id)
        span = f"""<a href="javascript:void(0)" class="token-span {' '.join(classes)}" id="token_{idx}" data-tooltip="{tooltip}">{safe_token}</a>"""
        token_spans.append(span)

    html_content = css + legend + f'<div class="token-stream-container"><div class="token-stream-text">{"".join(token_spans)}</div></div>'
    return html_content


def render_token_stream(token_data, selected_idx):
    """Wrapper for backward compatibility."""
    return render_token_stream_inline(token_data, selected_idx)
