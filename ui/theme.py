"""
iOS-inspired design system for the Streamlit UI.

Follows Apple's Human Interface Guidelines conventions: system font
(SF Pro on Apple devices), grouped-background + white "cell" cards,
large titles, segmented-control tabs, iMessage-style chat bubbles,
pill buttons, and the iOS system colour palette in light and dark mode.

Base colours come from .streamlit/config.toml ([theme.light]/[theme.dark]);
this module adds the component styling Streamlit's theme cannot express.
"""

PALETTES = {
    "light": {
        "bg": "#F2F2F7", "card": "#FFFFFF", "text": "#1C1C1E", "text2": "#6E6E73", "sep": "#E5E5EA",
        "fill": "rgba(118,118,128,0.12)", "blue": "#007AFF", "green": "#34C759", "orange": "#FF9500",
        "red": "#FF3B30", "purple": "#AF52DE", "indigo": "#5856D6", "teal": "#30B0C7", "gray": "#8E8E93",
        "shadow": "0 1px 2px rgba(0,0,0,0.04), 0 6px 20px rgba(0,0,0,0.05)",
        "bubble_ai": "#FFFFFF", "glass": "rgba(242,242,247,0.72)",
    },
    "dark": {
        "bg": "#000000", "card": "#1C1C1E", "text": "#F2F2F7", "text2": "#98989D", "sep": "#38383A",
        "fill": "rgba(118,118,128,0.24)", "blue": "#0A84FF", "green": "#30D158", "orange": "#FF9F0A",
        "red": "#FF453A", "purple": "#BF5AF2", "indigo": "#5E5CE6", "teal": "#40C8E0", "gray": "#8E8E93",
        "shadow": "0 1px 2px rgba(0,0,0,0.6)",
        "bubble_ai": "#1C1C1E", "glass": "rgba(0,0,0,0.6)",
    },
}


def palette(mode: str) -> dict:
    return PALETTES["dark" if mode == "dark" else "light"]


def css(mode: str = "light") -> str:
    p = palette(mode)
    return f"""
<style>
:root {{
  --ra-bg:{p['bg']}; --ra-card:{p['card']}; --ra-text:{p['text']}; --ra-text2:{p['text2']};
  --ra-sep:{p['sep']}; --ra-fill:{p['fill']}; --ra-blue:{p['blue']}; --ra-green:{p['green']};
  --ra-orange:{p['orange']}; --ra-red:{p['red']}; --ra-purple:{p['purple']}; --ra-indigo:{p['indigo']};
  --ra-teal:{p['teal']}; --ra-gray:{p['gray']}; --ra-shadow:{p['shadow']}; --ra-bubble-ai:{p['bubble_ai']};
}}
html, body, .stApp {{ -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }}
.block-container {{ max-width: 1160px; padding-top: 4.2rem; padding-bottom: 4rem; }}
header[data-testid="stHeader"] {{ background: {p['glass']}; backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px); }}

/* ---------- large title header ---------- */
.ra-hero {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:18px; }}
.ra-eyebrow {{ font-size:13px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--ra-blue); }}
.ra-title {{ font-size:34px; line-height:1.1; font-weight:700; letter-spacing:-0.022em; color:var(--ra-text); margin:2px 0 4px; }}
.ra-subtitle {{ font-size:15px; color:var(--ra-text2); max-width:640px; }}

/* ---------- widget cards (home-screen widgets) ---------- */
.ra-widgets {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:4px 0 22px; }}
@media (max-width: 820px) {{ .ra-widgets {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
.ra-widget {{ background:var(--ra-card); border-radius:18px; padding:14px 16px; box-shadow:var(--ra-shadow); min-width:0; }}
.ra-widget .lbl {{ display:flex; align-items:center; gap:8px; font-size:13px; font-weight:600; color:var(--ra-text2); }}
.ra-widget .val {{ font-size:28px; font-weight:700; letter-spacing:-0.02em; color:var(--ra-text); margin-top:6px; line-height:1.1; }}
.ra-widget .sub {{ font-size:12px; color:var(--ra-text2); margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

/* ---------- iOS settings-style icon squares ---------- */
.ra-icon {{ width:26px; height:26px; border-radius:7px; display:inline-flex; align-items:center; justify-content:center;
  color:#fff; font-size:14px; font-weight:700; flex:none; }}
.ra-icon.sm {{ width:22px; height:22px; border-radius:6px; font-size:12px; }}
.bg-blue {{ background:var(--ra-blue); }} .bg-green {{ background:var(--ra-green); }} .bg-orange {{ background:var(--ra-orange); }}
.bg-red {{ background:var(--ra-red); }} .bg-purple {{ background:var(--ra-purple); }} .bg-indigo {{ background:var(--ra-indigo); }}
.bg-teal {{ background:var(--ra-teal); }} .bg-gray {{ background:var(--ra-gray); }}

/* ---------- segmented-control tabs ---------- */
[data-testid="stTabs"] [role="tablist"] {{ background:var(--ra-fill); border-radius:11px; padding:3px; gap:2px;
  border:none !important; box-shadow:none; width:fit-content; max-width:100%; overflow-x:auto; }}
[data-testid="stTabs"] [role="tab"] {{ border-radius:8px !important; padding:6px 16px !important; margin:0 !important;
  font-weight:600; font-size:14px; border:none !important; background:transparent; color:var(--ra-text) !important;
  height:auto !important; white-space:nowrap; }}
[data-testid="stTabs"] [role="tab"] p {{ font-size:14px; font-weight:600; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ background:var(--ra-card) !important;
  box-shadow:0 1px 3px rgba(0,0,0,.12), 0 1px 1px rgba(0,0,0,.04); }}
[data-testid="stTabs"] .react-aria-SelectionIndicator, [data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display:none !important; }}
[data-testid="stTabs"] [role="tabpanel"] {{ padding-top:18px; }}

/* ---------- section headers (grouped list style) ---------- */
.ra-section {{ font-size:13px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:var(--ra-text2);
  margin:22px 4px 8px; }}
.ra-section:first-child {{ margin-top:4px; }}
.ra-h2 {{ font-size:22px; font-weight:700; letter-spacing:-0.015em; margin:26px 0 10px; color:var(--ra-text); }}

/* ---------- inset grouped list ---------- */
.ra-list {{ background:var(--ra-card); border-radius:14px; overflow:hidden; box-shadow:var(--ra-shadow); margin-bottom:14px; }}
.ra-row {{ display:flex; gap:12px; align-items:flex-start; padding:11px 16px; position:relative; }}
.ra-row + .ra-row::before {{ content:""; position:absolute; top:0; left:54px; right:0; border-top:0.5px solid var(--ra-sep); }}
.ra-row .body {{ min-width:0; flex:1; }}
.ra-row .title {{ font-weight:600; font-size:15px; color:var(--ra-text); overflow-wrap:anywhere; }}
[data-testid="stSidebar"] .ra-row .title {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.ra-row .mono {{ font-family:'SF Mono', ui-monospace, Menlo, monospace; font-size:13.5px; }}
.ra-row .sub {{ color:var(--ra-text2); font-size:13.5px; line-height:1.45; margin-top:1px; overflow-wrap:anywhere; }}
.ra-row .trail {{ margin-left:auto; padding-left:8px; color:var(--ra-text2); font-size:13px; white-space:nowrap; }}

/* ---------- pills / badges ---------- */
.ra-pills {{ display:flex; flex-wrap:wrap; gap:6px; margin:0 0 10px; }}
.ra-pill {{ display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:999px; font-size:12px;
  font-weight:600; background:var(--ra-fill); color:var(--ra-text2); white-space:nowrap; }}
.ra-pill.blue {{ background:color-mix(in srgb, var(--ra-blue) 14%, transparent); color:var(--ra-blue); }}
.ra-pill.green {{ background:color-mix(in srgb, var(--ra-green) 16%, transparent); color:{'#248A3D' if mode != 'dark' else p['green']}; }}
.ra-pill.orange {{ background:color-mix(in srgb, var(--ra-orange) 16%, transparent); color:{'#C93400' if mode != 'dark' else p['orange']}; }}
.ra-pill.red {{ background:color-mix(in srgb, var(--ra-red) 14%, transparent); color:var(--ra-red); }}
.ra-pill.purple {{ background:color-mix(in srgb, var(--ra-purple) 14%, transparent); color:var(--ra-purple); }}
.ra-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; flex:none; }}

/* ---------- chat: iMessage-style ---------- */
.ra-user-row {{ display:flex; justify-content:flex-end; margin:14px 0 8px; }}
.ra-user-bubble {{ background:var(--ra-blue); color:#fff; padding:9px 15px; border-radius:20px 20px 6px 20px;
  max-width:78%; font-size:15px; line-height:1.42; white-space:pre-wrap; overflow-wrap:anywhere; }}
[class*="st-key-answer"] {{ background:var(--ra-bubble-ai); border-radius:20px 20px 20px 6px; padding:14px 18px 8px;
  box-shadow:var(--ra-shadow); max-width:94%; }}
.ra-ai-label {{ display:flex; align-items:center; gap:8px; font-size:13px; font-weight:600; color:var(--ra-text2); margin-bottom:8px; }}

/* ---------- cards ---------- */
[class*="st-key-card"] {{ background:var(--ra-card); border-radius:16px; padding:16px 18px; box-shadow:var(--ra-shadow); }}
.ra-empty {{ background:var(--ra-card); border-radius:22px; padding:34px 28px; box-shadow:var(--ra-shadow); text-align:center; }}
.ra-empty .big {{ font-size:44px; line-height:1; margin-bottom:10px; }}
.ra-empty h3 {{ font-size:22px; font-weight:700; letter-spacing:-0.015em; margin:6px 0 6px; }}
.ra-empty p {{ color:var(--ra-text2); font-size:15px; margin:0 auto 18px; max-width:520px; }}
.ra-steps {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; text-align:left; max-width:720px; margin:0 auto; }}
@media (max-width: 720px) {{ .ra-steps {{ grid-template-columns:1fr; }} }}
.ra-step {{ background:var(--ra-fill); border-radius:14px; padding:12px 14px; }}
.ra-step b {{ display:block; font-size:14px; margin:6px 0 2px; color:var(--ra-text); }}
.ra-step .desc {{ font-size:13px; color:var(--ra-text2); }}

/* ---------- forms on cards (iOS grouped text fields) ---------- */
[class*="st-key-card"] [data-testid="stForm"] {{ border:none !important; padding:4px 0 0 !important; }}
[class*="st-key-card"] [data-testid="stTextInputRootElement"], [class*="st-key-card"] [data-baseweb="input"] {{
  background:var(--ra-fill) !important; border-radius:10px !important; border-color:transparent !important; }}
[class*="st-key-card"] input {{ background:transparent !important; font-size:15px; }}
[data-testid="stTextInputRootElement"]:focus-within {{ box-shadow:0 0 0 3px color-mix(in srgb, var(--ra-blue) 35%, transparent); }}

/* ---------- motion (iOS-style springs; disabled for "reduce motion") ---------- */
@keyframes raFadeUp {{ from {{ opacity:0; transform:translateY(12px) scale(.985); }} to {{ opacity:1; transform:none; }} }}
@keyframes raFadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
@keyframes raPopRight {{ 0% {{ opacity:0; transform:translateY(10px) scale(.85); }} 70% {{ opacity:1; transform:scale(1.02); }} 100% {{ transform:none; }} }}
@keyframes raPopLeft {{ 0% {{ opacity:0; transform:translateY(12px) scale(.94); }} 100% {{ opacity:1; transform:none; }} }}
@keyframes raPulse {{ 0%,100% {{ box-shadow:0 0 0 0 color-mix(in srgb, var(--ra-green) 55%, transparent); }} 60% {{ box-shadow:0 0 0 6px transparent; }} }}
@keyframes raDot {{ 0%,80%,100% {{ transform:translateY(0); opacity:.35; }} 40% {{ transform:translateY(-5px); opacity:1; }} }}
@keyframes raFlow {{ from {{ background-position:0% 50%; }} to {{ background-position:300% 50%; }} }}
@keyframes raShine {{ from {{ background-position:120% 0; }} to {{ background-position:-20% 0; }} }}

.ra-hero, .ra-empty, .ra-list, [class*="st-key-card"], [data-testid="stExpander"] {{ animation:raFadeUp .5s cubic-bezier(.2,.8,.2,1) both; }}
.ra-widget {{ animation:raFadeUp .55s cubic-bezier(.2,.8,.2,1) both; transition:transform .25s cubic-bezier(.2,.8,.2,1), box-shadow .25s; }}
.ra-widget:nth-child(2) {{ animation-delay:.06s; }} .ra-widget:nth-child(3) {{ animation-delay:.12s; }} .ra-widget:nth-child(4) {{ animation-delay:.18s; }}
.ra-widget:hover {{ transform:translateY(-3px); box-shadow:0 2px 4px rgba(0,0,0,.05), 0 14px 32px rgba(0,0,0,.09); }}
.ra-row {{ animation:raFadeUp .45s cubic-bezier(.2,.8,.2,1) both; transition:background .2s; }}
.ra-row:nth-child(2) {{ animation-delay:.04s; }} .ra-row:nth-child(3) {{ animation-delay:.08s; }} .ra-row:nth-child(4) {{ animation-delay:.12s; }}
.ra-row:nth-child(5) {{ animation-delay:.16s; }} .ra-row:nth-child(6) {{ animation-delay:.2s; }} .ra-row:nth-child(n+7) {{ animation-delay:.24s; }}
.ra-row:hover {{ background:color-mix(in srgb, var(--ra-fill) 60%, transparent); }}
.ra-user-bubble {{ animation:raPopRight .42s cubic-bezier(.34,1.56,.64,1) both; transform-origin:bottom right; }}
[class*="st-key-answer"] {{ animation:raPopLeft .45s cubic-bezier(.2,.8,.2,1) both; transform-origin:bottom left; }}
[data-testid="stTabs"] [role="tabpanel"] {{ animation:raFadeIn .3s ease both; }}
.ra-pill {{ transition:transform .2s; }} .ra-pill:hover {{ transform:scale(1.04); }}
.ra-live {{ width:8px; height:8px; border-radius:50%; background:var(--ra-green); display:inline-block; animation:raPulse 2s ease-out infinite; }}
.ra-title {{ background:linear-gradient(100deg, var(--ra-text) 40%, var(--ra-blue) 50%, var(--ra-purple) 55%, var(--ra-text) 65%);
  background-size:250% 100%; -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  animation:raShine 2.2s cubic-bezier(.4,0,.2,1) .2s 1 both; }}

/* typing indicator (iMessage) */
.ra-typing {{ display:inline-flex; gap:5px; align-items:center; background:var(--ra-bubble-ai); padding:12px 16px;
  border-radius:20px 20px 20px 6px; box-shadow:var(--ra-shadow); margin:4px 0 10px; animation:raPopLeft .3s ease both; }}
.ra-typing span {{ width:8px; height:8px; border-radius:50%; background:var(--ra-gray); animation:raDot 1.2s infinite ease-in-out; }}
.ra-typing span:nth-child(2) {{ animation-delay:.15s; }} .ra-typing span:nth-child(3) {{ animation-delay:.3s; }}

/* Apple-Intelligence-style glow around the agent while it works */
[class*="st-key-thinking"] {{ position:relative; border-radius:18px; padding:2.5px !important; isolation:isolate;
  background:linear-gradient(90deg,#007AFF,#AF52DE,#FF2D55,#FF9500,#FFCC00,#34C759,#5AC8FA,#007AFF);
  background-size:300% 100%; animation:raFlow 4s linear infinite; }}
[class*="st-key-thinking"]::before {{ content:""; position:absolute; inset:-6px; border-radius:24px; z-index:-1;
  background:inherit; background-size:300% 100%; animation:raFlow 4s linear infinite; filter:blur(16px); opacity:.55; }}
[class*="st-key-thinking"] [data-testid="stExpander"] details, [class*="st-key-thinking"] [data-testid="stStatusWidget"] {{
  background:var(--ra-card) !important; border-radius:16px !important; }}

/* galaxy */
[class*="st-key-galaxy"] {{ background:var(--ra-card); border-radius:22px; padding:14px 16px 6px; box-shadow:var(--ra-shadow);
  animation:raFadeUp .6s cubic-bezier(.2,.8,.2,1) both; }}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation:none !important; transition:none !important; }}
  .ra-title {{ -webkit-text-fill-color:var(--ra-text); background:none; }}
}}

/* ---------- Streamlit widgets ---------- */
[data-testid="stExpander"] details {{ border:none !important; border-radius:14px !important; background:var(--ra-card);
  box-shadow:var(--ra-shadow); }}
[data-testid="stExpander"] summary {{ font-weight:600; }}
[data-testid="stSidebar"] [data-testid="stExpander"] details {{ background:var(--ra-bg); box-shadow:none; }}
[data-testid="stChatInput"] {{ border-radius:22px !important; box-shadow:var(--ra-shadow); }}
[data-testid="stChatInput"] textarea {{ font-size:15px; }}
[data-testid="stAlert"] {{ border-radius:14px; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius:12px; overflow:hidden; }}
[data-testid="stStatusWidget"] {{ border-radius:14px; }}
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] {{ padding-top:1rem; }}
.ra-brand {{ display:flex; align-items:center; gap:10px; margin:0 0 6px; }}
.ra-brand .name {{ font-size:19px; font-weight:700; letter-spacing:-0.01em; }}
.ra-brand .ver {{ font-size:12px; color:var(--ra-text2); }}
.ra-muted {{ color:var(--ra-text2); font-size:13px; }}
.ra-quote {{ background:var(--ra-fill); border-radius:12px; padding:10px 12px; font-size:13.5px; line-height:1.5;
  color:var(--ra-text); white-space:pre-wrap; overflow-wrap:anywhere; max-height:220px; overflow:auto; }}
</style>
"""
