import textwrap

def render_token_card(token, index, prob, is_critical, reason, is_selected, is_new=False):
    classes = ["token-card"]
    if is_critical:
        classes.append(f"critical-{reason}")
    if is_selected:
        classes.append("selected")
    if is_new:
        classes.append("new-token")
    
    safe_token = token.replace("<", "&lt;").replace(">", "&gt;")
    
    # Structure: Main token text + Sub info (Index | Prob)
    content = f"""
        <div class="token-main">{safe_token}</div>
        <div class="token-sub">{index} <span style="color:#ccc">|</span> {prob:.2f}</div>
    """
    
    # Use 'id' for st-click-detector if available, otherwise fallback to link
    # We'll generate an anchor with an ID that st-click-detector can catch
    # href='#' prevents navigation if caught by JS, but st-click-detector handles it.
    return f"""<a href='#' id='token_{index}' class="{' '.join(classes)}">{content}</a>"""


def render_token_stream(token_data, selected_idx):
    css = textwrap.dedent("""
    <style>
        .token-card {
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            vertical-align: middle;
            margin: 3px;
            padding: 4px 8px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            background-color: #ffffff;
            color: #333;
            text-decoration: none !important;
            font-family: 'Source Code Pro', monospace;
            transition: all 0.2s ease;
            min-width: 40px;
        }
        .token-card:hover {
            border-color: #b0b0b0;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            color: #000;
        }
        .token-main {
            font-size: 14px;
            font-weight: 600;
            line-height: 1.2;
        }
        .token-sub {
            font-size: 10px;
            color: #888;
            margin-top: 2px;
            line-height: 1;
        }
        
        /* Critical States */
        .token-card.critical-divergence {
            background-color: #fff5f5;
            border-color: #ffcdd2;
        }
        .token-card.critical-divergence .token-sub { color: #e57373; }
        
        .token-card.critical-low_prob {
            background-color: #fffde7;
            border-color: #fff9c4;
        }
        .token-card.critical-low_prob .token-sub { color: #fbc02d; }
        
        .token-card.critical-high_entropy {
            background-color: #e8f5e9;
            border-color: #c8e6c9;
        }
        
        /* Selected State */
        .token-card.selected {
            border-color: #2196f3;
            background-color: #e3f2fd;
            box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.3);
        }
        .token-card.selected .token-main { color: #1565c0; }
        .token-card.selected .token-sub { color: #1976d2; }

        /* New Token State */
        .token-card.new-token .token-main {
            text-decoration: underline;
            text-decoration-color: #2196f3;
            text-decoration-thickness: 2px;
        }
        
    </style>
    """)
    
    html_content = css + '<div style="line-height: 2.5; padding: 10px;">'
    for item in token_data:
        html_content += render_token_card(
            item['token'], 
            item['index'], 
            item['prob'], 
            item['is_critical'], 
            item['reason'], 
            item['index'] == selected_idx,
            item.get('is_new', False)
        )
    html_content += "</div>"
    return html_content
