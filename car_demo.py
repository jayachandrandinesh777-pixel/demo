"""
Animated car demo (Streamlit).  Run:  streamlit run car_demo.py
Move the slider to see the car change colour, speed and badge with the MPG value.
"""
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Car animation demo", page_icon="🚗")
st.title("🚗 Animated car demo")

CAR_HTML = """
<style>
  html,body{margin:0;background:transparent;font-family:system-ui,sans-serif}
  .scene{position:relative;height:200px;border-radius:14px;overflow:hidden;
         background:linear-gradient(#7dd3fc,#e0f2fe 70%)}
  .hills{position:absolute;left:0;bottom:48px;width:200%;height:70px;opacity:.55;
         background:radial-gradient(60px 40px at 60px 70px,#4ade80 98%,transparent) 0 0/180px 70px repeat-x;
         animation:scroll calc(var(--t)*5) linear infinite}
  .cloud{position:absolute;top:22px;left:0;width:200%;height:40px;opacity:.9;
         background:radial-gradient(28px 14px at 40px 20px,#fff 95%,transparent) 0 0/260px 40px repeat-x;
         animation:scroll calc(var(--t)*9) linear infinite}
  .road{position:absolute;left:0;right:0;bottom:0;height:52px;background:#374151}
  .lane{position:absolute;left:0;bottom:22px;width:200%;height:5px;
        background:repeating-linear-gradient(90deg,#fde68a 0 36px,transparent 36px 72px);
        animation:scroll var(--t) linear infinite}
  .car{position:absolute;left:50%;bottom:34px;width:210px;margin-left:-105px;
       animation:bob .5s ease-in-out infinite alternate}
  .wheel{transform-box:fill-box;transform-origin:center;animation:spin calc(var(--t)*.6) linear infinite}
  .smoke circle{fill:#9ca3af;opacity:0;animation:puff 1.2s ease-out infinite}
  .smoke circle:nth-child(2){animation-delay:.4s}.smoke circle:nth-child(3){animation-delay:.8s}
  .badge{position:absolute;top:10px;right:12px;background:#0008;color:#fff;padding:6px 12px;
         border-radius:20px;font-size:13px;font-weight:600}
  .bar{position:absolute;left:12px;top:12px;width:120px;height:10px;background:#0003;border-radius:6px}
  .bar i{display:block;height:100%;width:var(--fill);background:var(--c);border-radius:6px;transition:width 1s}
  @keyframes scroll{to{transform:translateX(-50%)}}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes bob{to{transform:translateY(-3px)}}
  @keyframes puff{0%{opacity:.7;transform:translate(0,0) scale(.5)}100%{opacity:0;transform:translate(-40px,-14px) scale(1.6)}}
</style>
<div class="scene" style="--t:__SPEED__s;--c:__COLOR__;--fill:__FILL__%">
  <div class="cloud"></div><div class="hills"></div>
  <div class="bar"><i></i></div><div class="badge">__LABEL__</div>
  <div class="road"><div class="lane"></div></div>
  <svg class="car" viewBox="0 0 210 80" xmlns="http://www.w3.org/2000/svg">
    <g class="smoke"><circle cx="10" cy="56" r="6"/><circle cx="10" cy="56" r="6"/><circle cx="10" cy="56" r="6"/></g>
    <path d="M14 56 L14 44 Q14 38 22 36 L52 32 L74 12 Q78 8 86 8 L132 8 Q140 8 146 14 L164 32 L190 38 Q198 40 198 48 L198 56 Z" fill="__COLOR__"/>
    <path d="M80 15 L118 15 L118 32 L58 32 Z" fill="#bae6fd"/>
    <path d="M124 15 L136 15 Q140 15 143 19 L156 32 L124 32 Z" fill="#bae6fd"/>
    <rect x="186" y="42" width="12" height="7" rx="3" fill="#fde047"/>
    <rect x="14" y="44" width="8" height="6" rx="2" fill="#ef4444"/>
    <g><circle cx="55" cy="58" r="17" fill="#111827"/>
      <g class="wheel"><circle cx="55" cy="58" r="9" fill="#d1d5db"/>
        <path d="M55 49V67M46 58H64" stroke="#6b7280" stroke-width="3"/></g></g>
    <g><circle cx="155" cy="58" r="17" fill="#111827"/>
      <g class="wheel"><circle cx="155" cy="58" r="9" fill="#d1d5db"/>
        <path d="M155 49V67M146 58H164" stroke="#6b7280" stroke-width="3"/></g></g>
  </svg>
</div>
"""


def car_html(mpg=None):
    """Animated car. After a prediction its colour/speed reflect fuel efficiency."""
    if mpg is None:
        color, speed, fill, label = "#3b82f6", 1.4, 0, "Enter specs and predict ➜"
    elif mpg >= 30:
        color, speed, fill, label = "#10b981", 1.0, min(mpg / 50, 1) * 100, f"{mpg:.1f} MPG · Efficient 🌿"
    elif mpg >= 20:
        color, speed, fill, label = "#f59e0b", 1.5, min(mpg / 50, 1) * 100, f"{mpg:.1f} MPG · Average"
    else:
        color, speed, fill, label = "#ef4444", 2.2, min(mpg / 50, 1) * 100, f"{mpg:.1f} MPG · Thirsty ⛽"
    return (CAR_HTML.replace("__COLOR__", color).replace("__SPEED__", str(speed))
            .replace("__FILL__", f"{fill:.0f}").replace("__LABEL__", label))


predicted = st.checkbox("Show a prediction", value=True)
mpg = st.slider("Predicted MPG", 5, 50, 25)
components.html(car_html(mpg if predicted else None), height=215)
st.caption("Green = 30+ MPG (efficient), amber = 20-30 (average), red = under 20 (thirsty).")
