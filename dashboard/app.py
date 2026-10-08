import os 
import pandas as pd 
import plotly.express as px
import requests
import streamlit as st 

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
COLORS = {"news": "#2563eb", "reddit": "#f97316"}
SENT_COLORS = {"Positive": "#16a34a", "Neutral": "#9ca3af", "Negative": "#dc2626"}

st.set_page_config(page_title="CrowdPulse", page_icon="📡", layout="wide")



class NotReady(Exception):
    """API answered 503: analysis not available yet."""


@st.cache_data(ttl=30, show_spinner=False)
def _cached_get(path: str, params: dict | None = None):
    r = requests.get(
        f"{API}{path}",
        params=params,
        timeout=30,
    )
    if r.status_code == 503:
        raise NotReady 

    r.raise_for_status()
    return r.json()

def api_get(path: str , params : dict | None = None): 
    try:
        return _cached_get(path , params)
    except NotReady:
        return None
api_get.clear = _cached_get.clear


def _error_detail(r : requests.Response) -> str:
    try:
        return r.json().get("detail" , r.text)
    except ValueError:
        return r.text or f"HTTP {r.status_code}"
    
    
def api_post(path: str, payload: dict, timeout: int = 120):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=timeout)

    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not reach the API: {e}"
        ) from e
    if not r.ok:
        raise RuntimeError(_error_detail(r))

    return r.json()



def pair_label(p: dict) -> str:
    return f"{p['news']['name']}  ⇄  {p['reddit']['name']}"

try:
    status = requests.get(f"{API}/scraper/status", timeout=5).json()
except (requests.RequestException , ValueError):
    st.error(f"Can't reach the CrowdPulse API at {API}. Start it with: `uvicorn api.main:app --port 8000`")
    st.stop()


with st.sidebar:
    st.title("📡 CrowdPulse")
    st.caption("How do News and Reddit frame the same event differently?")
    st.divider()
    st.caption(f"Pipeline: **{status['state']}**" + (f" · {status['step']}" if status["state"] == "running" else ""))
    if status.get("error"):
        st.error(status["error"])
    if st.button("🔄 Scrape & re-analyse", disabled=status["state"] == "running", width="stretch"):
        try:
            api_post("/scraper/run", {})
            api_get.clear()
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.button("Reload dashboard", width="stretch"):
        api_get.clear()
        st.rerun()

summary = api_get("/data/summary")
if summary is None:
    st.info("⏳ No analysis yet. The first pipeline run (scrape → NLP → save) is in progress; "
            "this can take a few minutes the first time while models download. Reload shortly.")
    st.stop()

sent , counts = summary['sentiment'] , summary['counts']
st.sidebar.caption(f"Last updated: {summary['generated_at'][:16].replace('T', ' ')} UTC")

gap = sent["reddit"]["avg"] - sent["news"]["avg"]
c1 , c2 , c3 , c4 = st.columns(4)
c1.metric("News Articles" , counts["news"])
c2.metric("Reddit Post" , counts["reddit"])
c3.metric("Avg sentiment · News", f"{sent['news']['avg']:+.2f}")
c4.metric("Avg sentiment · Reddit", f"{sent['reddit']['avg']:+.2f}", f"{gap:+.2f} vs News", delta_color="off")

tab_over, tab_story, tab_ent, tab_time, tab_task = st.tabs(
    [
        "Overview",
        "Same event, different story",
        "Entities",
        "Timeline",
        "Ask CrowdPulse",
    ]
)


with tab_over:
    st.subheader("Sentiment")
    mix = pd.DataFrame([{"Source" : s.title() , "Sentiment":lab.title() , "Share (%)" : sent[s][lab]}
                        for s in ("news" , "reddit") for lab in ("positive" , "neutral" , "negative")])

    fig = px.bar(mix, x="Share (%)", y="Source", color="Sentiment", orientation="h",
                 color_discrete_map=SENT_COLORS, text_auto=".0f", height=240)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
    st.plotly_chart(fig, width="stretch")

    st.subheader("What each source is talking about")
    topics = api_get("/topics") or {"news" : [] , "reddit" : []}
    left , right = st.columns(2)
    for col , src in ((left , "news") , (right , "reddit")):
        with col:
            st.markdown(f"**{src.title()}**")
            df = pd.DataFrame(topics[src])
            if df.empty:
                st.caption("Not enough data for topic modeling yet")
                continue
            fig = px.bar(df, x="count", y="name", orientation="h", color="avg_sentiment",
                         color_continuous_scale="RdYlGn", range_color=(-1, 1), height=380,
                         labels={"count": "Items", "name": "", "avg_sentiment": "Sentiment"})
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")


def render_narrative(n : dict , pair : dict | None):
    if pair:
        a , b , c = st.columns(3)
        a.metric("News sentiment", f"{pair['news']['avg_sentiment']:+.2f}", f"{pair['news']['count']} articles", delta_color="off")
        b.metric("Reddit sentiment", f"{pair['reddit']['avg_sentiment']:+.2f}", f"{pair['reddit']['count']} posts", delta_color="off")
        c.metric("Topic similarity (cosine)", f"{pair['score']:.2f}")
    l, r = st.columns(2)
    l.markdown("**How news frames it**")
    l.write(n["news_framing"])
    r.markdown("**💬 How Reddit frames it**")
    r.write(n["reddit_framing"])

    st.info(f"**Key difference:** {n['key_difference']}")
    if n.get("generated_by") == "fallback":
        st.caption("Statistical summary only: set GROQ_API_KEY for the Llama 3.3 narrative.")
    if pair:
        s1, s2 = st.columns(2)
        with s1:
            st.caption("Representative news items")
            for s in pair["news"]["samples"][:3]:
                st.markdown(f"- [{s['title'][:110]}]({s['url']}) · *{s['origin']}*")
        with s2:
            st.caption("Representative Reddit posts")
            for s in pair["reddit"]["samples"][:3]:
                st.markdown(f"- [{s['title'][:110]}]({s['url']}) · *{s['origin']}*")

with tab_story:
    pairs = api_get("/similarity") or []
    narratives = api_get("/narrative") or []

    if not pairs:
        st.info(
            "No News ⇄ Reddit topic pairs matched yet "
            "(needs >= 30 items per source)."
        )

    else:
        st.subheader(
            "Sentiment gap per shared topic (Reddit − News)"
        )

        gdf = pd.DataFrame(
            [
                {
                    "Topic": pair_label(p),
                    "Gap": p["sentiment_gap"],
                }
                for p in pairs
            ]
        )

        fig = px.bar(gdf, x="Gap", y="Topic", orientation="h", color="Gap",
            color_continuous_scale="RdBu", range_color=(-1, 1),height=max(220, 50 * len(gdf)),
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.caption(
            "Positive = Reddit is more positive than the "
            "news on that topic; negative = more negative."
        )

        st.subheader("Narrative comparison")

        by_key = {
            (
                p["news_topic"],
                p["reddit_topic"],
            ): p
            for p in pairs
        }

        for i, n in enumerate(narratives):
            with st.expander(
                f"📌 {n['event']}",
                expanded=(i == 0),
            ):
                render_narrative(
                    n,
                    by_key.get(
                        (
                            n["news_topic"],
                            n["reddit_topic"],
                        )
                    ),
                )
        st.subheader("Compare another topic pair")

        choice = st.selectbox(
            "Pick a pair",
            range(len(pairs)),
            format_func=lambda i: pair_label(
                pairs[i]
            ),
        )

        if st.button("Generate narrative"):
            with st.spinner("Asking the model…"):
                try:
                    p = pairs[choice]

                    result = api_post(
                        "/narrative",
                        {
                            "news_topic": p["news_topic"],
                            "reddit_topic": p["reddit_topic"],
                        },
                    )
                    st.session_state["custom_narrative"] = (
                        p,
                        result,
                    )
                except RuntimeError as e:
                    st.session_state.pop(
                        "custom_narrative",
                        None,
                    )
                    st.error(e)
        saved = st.session_state.get("custom_narrative")

        if saved:
            st.markdown(
                f"**{pair_label(saved[0])}**"
            )
            render_narrative(
                saved[1],
                saved[0],
            )

with tab_ent:
    ents = pd.DataFrame(api_get("/entities") or [])
    if ents.empty:
        st.info("No entities yet.")
    else:
        st.subheader("Who gets mentioned where")
        st.caption("Share = % of that source's items that mention the entity (fair across different volumes).")
        top = ents.head(15).melt(id_vars="entity", value_vars=["news_share", "reddit_share"],
                                 var_name="Source", value_name="Share (%)")
        top["Source"] = top["Source"].str.replace("_share", "")
        fig = px.bar(top, x="entity", y="Share (%)", color="Source", barmode="group",
                     color_discrete_map=COLORS, height=380)
        fig.update_layout(xaxis_title=None, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Sentiment around each entity")
        st.dataframe(
            ents[["entity", "news_count", "reddit_count", "news_sentiment", "reddit_sentiment"]]
            .rename(columns={"news_count": "News mentions", "reddit_count": "Reddit mentions",
                             "news_sentiment": "News sentiment", "reddit_sentiment": "Reddit sentiment"}),
            width="stretch", hide_index=True)

with tab_time:
    tl = api_get("/data/timeline") or {"daily" : [] , "topic_volume" : []}
    daily = pd.DataFrame(tl["daily"])
    if daily.empty:
        st.info("No timeline data yet.")
    else:
        st.subheader("Average sentiment by day")
        fig = px.line(daily , x="date" , y = "avg_sentiment" , color= "source" , markers=True , color_discrete_map=COLORS , height=320,
                      labels={"avg_sentiment" : "Sentiment"})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Topic volume by day")
        src = st.radio("Source", ["news", "reddit"], horizontal=True)
        vol = pd.DataFrame(tl["topic_volume"])
        vol = vol[vol["source"] == src] if not vol.empty else vol
        if vol.empty:
            st.caption("No topic data for this source.")
        else:
            fig = px.bar(vol, x="date", y="docs", color="topic_name", barmode="stack", height=360,
                         labels={"docs": "Items", "topic_name": "Topic"})
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")

with tab_task:
    st.subheader("Ask about the data")
    st.session_state.setdefault("history" , [])
    with st.form("ask" , clear_on_submit=True):
        question = st.text_input("Question" , placeholder="How do News and Reddit differ on the economy?")
        submitted = st.form_submit_button("Ask")
    if submitted and question.strip():
        with st.spinner("Retrieving sources and thinking.."):
            try:
                st.session_state.history.insert(0 , {"q" : question , **api_post("/qa" , {"question" : question})})

            except RuntimeError as e:
                st.error(str(e))


    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["q"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            with st.expander(f"Sources ({len(item['sources'])})"):
                for s in item["sources"]:
                    tag = "📰" if s["source"] == "news" else "💬"
                    st.markdown(f"**[{s['n']}]** {tag} [{s['title'][:120]}]({s['url']}) · *{s['origin']}*")